#!/usr/bin/env python3
import os, argparse
import numpy as np
from sklearn.utils import shuffle

# ── TensorFlow / Supervised Phase ──
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    Dense, Bidirectional, LSTM, GlobalAveragePooling1D
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping

# ── SB3 / RL² Phase ──
import gym, torch, torch.nn as nn
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy


# — Data loading & sequence builder —
def create_lidar_sequences(lidar, servo, speed, ts, seq_len=5):
    X, y = [], []
    N = lidar.shape[1]
    for i in range(len(lidar) - seq_len):
        frames = lidar[i : i+seq_len]
        dt = np.diff(ts[i : i+seq_len+1]).reshape(seq_len,1)
        dt = np.repeat(dt, N, axis=1)
        X.append(np.stack([frames, dt], axis=2))
        y.append([servo[i+seq_len], speed[i+seq_len]])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def load_all(npz_paths, seq_len):
    all_l, all_s, all_sp, all_t = [], [], [], []
    for p in npz_paths:
        d = np.load(p)
        all_l.append(d['lidar']); all_s.append(d['servo'])
        all_sp.append(d['speed']); all_t.append(d['ts'])
    L = np.concatenate(all_l, axis=0)
    S = np.concatenate(all_s, axis=0)
    P = np.concatenate(all_sp, axis=0)
    T = np.concatenate(all_t, axis=0)
    return create_lidar_sequences(L, S, P, T, seq_len)

# — Supervised model definition —
def build_supervised(seq_len, num_ranges):
    inp = Input((seq_len, num_ranges, 2), name='in')
    x = TimeDistributed(Conv1D(64,10,4,activation='relu'))(inp)
    x = TimeDistributed(Conv1D(128,8,4,activation='relu'))(x)
    x = TimeDistributed(Conv1D(256,4,2,activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(2, activation='tanh')(x)
    return Model(inp, out, name='sup_net')

# — RL² policy & extractor —
class LidarExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, hidden_size=64):
        super().__init__(obs_space, features_dim=hidden_size)
        seq_len, num_ranges, ch = obs_space.shape
        self.conv = nn.Sequential(
            nn.Conv1d(ch*seq_len,64,10,4), nn.ReLU(),
            nn.Conv1d(64,128,8,4),      nn.ReLU(),
            nn.Conv1d(128,256,4,2),     nn.ReLU(),
        )
        self.lstm = nn.LSTM(256,128,bidirectional=True,batch_first=True)
        self.fc   = nn.Linear(128*2, hidden_size)

    def forward(self, x):
        b,T,N,C = x.shape
        x = x.permute(0,3,1,2).reshape(b, C*T, N)
        x = self.conv(x).permute(0,2,1)
        out,_ = self.lstm(x)
        out = out.mean(1)
        return self.fc(out)

class RecurrentLidarPolicy(RecurrentActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
            policy_kwargs=dict(
                features_extractor_class=LidarExtractor,
                features_extractor_kwargs=dict(hidden_size=64)
            ), **kwargs)

# — Main training pipeline —
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',            nargs='+', required=True)
    p.add_argument('--batch_size',     type=int, default=64)
    p.add_argument('--sup_epochs',     type=int, default=100)
    p.add_argument('--rl_timesteps',   type=int, default=500_000)
    p.add_argument('--output_supervised', default='Models/supervised.h5')
    p.add_argument('--output_rl',         default='Models/ppo_rln_rl2.zip')
    opts = p.parse_args()

    seq_len = 5
    # — Supervised phase —
    X, y = load_all(opts.npz, seq_len)
    X, y = shuffle(X, y, random_state=42)
    split = int(0.85 * len(X))
    X_tr, y_tr = X[:split], y[:split]
    X_va, y_va = X[split:], y[split:]

    # normalize
    mu = X_tr.mean((0,1), keepdims=True)
    sd = X_tr.std((0,1), keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd

    # build, compile, schedule
    model = build_supervised(seq_len, X.shape[2])
    steps = len(X_tr) // opts.batch_size
    total = steps * opts.sup_epochs
    lr_schedule = CosineDecay(5e-4, decay_steps=total, alpha=1e-3)
    model.compile(Adam(learning_rate=lr_schedule), loss='mse')


    train_ds = (
        tf.data.Dataset
          .from_tensor_slices((X_tr, y_tr))
          .shuffle(2000)
          .batch(opts.batch_size)
          # graph-based noise injection preserves shape!
          .map(lambda A, B: (A + tf.random.normal(tf.shape(A), 0.0, 0.01), B),
               num_parallel_calls=tf.data.AUTOTUNE)
          .prefetch(tf.data.AUTOTUNE)
    )

    cb = [EarlyStopping('val_loss', patience=10, restore_best_weights=True)]
    hist = model.fit(train_ds,
                     validation_data=(X_va, y_va),
                     epochs=opts.sup_epochs,
                     callbacks=cb,
                     verbose=2)

    os.makedirs(os.path.dirname(opts.output_supervised), exist_ok=True)
    model.save(opts.output_supervised)
    print("Supervised best val_loss:", min(hist.history['val_loss']))

    # — RL² fine-tuning phase —
    # now define your real env:
    class LidarSequenceEnv(gym.Env):
        def __init__(self, sequences, targets):
            super().__init__()
            self.sequences = sequences
            self.targets   = targets
            self.idx       = 0
            obs_shape = sequences.shape[1:]
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
            )
            self.action_space = gym.spaces.Box(
                low=-1.0, high=1.0, shape=(2,), dtype=np.float32
            )

        def reset(self):
            self.idx = 0
            return self.sequences[self.idx]

        def step(self, action):
            target = self.targets[self.idx]
            reward = -np.mean((action - target) ** 2)
            self.idx += 1
            done = self.idx >= len(self.sequences)
            obs = (self.sequences[self.idx]
                   if not done else
                   np.zeros_like(self.sequences[0]))
            return obs, reward, done, {}

    env = DummyVecEnv([lambda: LidarSequenceEnv(X_tr, y_tr)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    rl = RecurrentPPO(
        policy=RecurrentLidarPolicy,
        env=env,
        learning_rate=5e-5,
        n_steps=seq_len * 20,
        batch_size=256,
        n_epochs=10,
        gamma=0.995,
        tensorboard_log='rl2_tb/',
        verbose=1,
        device='cuda'
    )
    rl.learn(total_timesteps=opts.rl_timesteps)
    os.makedirs(os.path.dirname(opts.output_rl), exist_ok=True)
    rl.save(opts.output_rl)
    print("RL² policy saved to", opts.output_rl)

if __name__ == '__main__':
    main()


# #!/usr/bin/env python3
# import os
# import time
# import warnings
# import numpy as np
# import tensorflow as tf
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# from sklearn.utils import shuffle
# import gym

# # PyTorch, SB3, and SB3-Contrib imports
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from stable_baselines3.common.vec_env import DummyVecEnv
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from sb3_contrib import RecurrentPPO
# from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

# #========================================================
# # Utility functions
# #========================================================
# def linear_map(x, x_min, x_max, y_min, y_max):
#     return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


# def create_lidar_sequences(lidar_data, servo_data, speed_data, timestamps, sequence_length=5):
#     X, y = [], []
#     num_ranges = lidar_data.shape[1]
#     for i in range(len(lidar_data) - sequence_length):
#         frames = np.stack(lidar_data[i : i + sequence_length], axis=0)
#         dt = np.diff(timestamps[i : i + sequence_length + 1]).reshape(sequence_length, 1)
#         dt_tiled = np.repeat(dt, num_ranges, axis=1)
#         seq = np.concatenate([frames[..., None], dt_tiled[..., None]], axis=2)
#         X.append(seq)
#         y.append([servo_data[i + sequence_length], speed_data[i + sequence_length]])
#     return np.array(X), np.array(y)


# def load_all_data(npz_paths, seq_len):
#     all_lidar, all_servo, all_speed, all_ts = [], [], [], []
#     for p in npz_paths:
#         data = np.load(p)
#         all_lidar.append(data['lidar'])
#         all_servo.append(data['servo'])
#         all_speed.append(data['speed'])
#         all_ts.append(data['ts'])
#     lidar = np.concatenate(all_lidar, axis=0)
#     servo = np.concatenate(all_servo, axis=0)
#     speed = np.concatenate(all_speed, axis=0)
#     ts    = np.concatenate(all_ts, axis=0)
#     return create_lidar_sequences(lidar, servo, speed, ts, sequence_length=seq_len)

# #========================================================
# # Supervised spatiotemporal model (TF)
# #========================================================
# from tensorflow.keras.layers import (
#     Input, TimeDistributed, Conv1D, Flatten,
#     Bidirectional, LSTM, Dense, Attention, GlobalAveragePooling1D
# )
# from tensorflow.keras.models import Model
# from tensorflow.keras.optimizers import Adam

# def build_spatiotemporal_model(seq_len, num_ranges):
#     inp = Input(shape=(seq_len, num_ranges, 2), name='lidar_sequence')
#     x = TimeDistributed(Conv1D(24, 10, strides=4, activation='relu'))(inp)
#     x = TimeDistributed(Conv1D(36, 8, strides=4, activation='relu'))(x)
#     x = TimeDistributed(Conv1D(48, 4, strides=2, activation='relu'))(x)
#     x = TimeDistributed(Flatten())(x)
#     lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
#     q = Dense(64)(lstm_out)
#     k = Dense(64)(lstm_out)
#     v = Dense(64)(lstm_out)
#     attn = Attention()([q, v, k])
#     context = GlobalAveragePooling1D()(attn)
#     out = Dense(2, activation='tanh', name='controls')(context)
#     return Model(inp, out, name='RNN_Attention_Controller')

# #========================================================
# # Custom PyTorch feature extractor for SB3
# #========================================================
# class LidarExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, hidden_size=64):
#         super().__init__(observation_space, features_dim=hidden_size)
#         seq_len, num_ranges, channels = observation_space.shape
#         self.conv = nn.Sequential(
#             nn.Conv1d(channels * seq_len, 24, kernel_size=10, stride=4),
#             nn.ReLU(),
#             nn.Conv1d(24, 36, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv1d(36, 48, kernel_size=4, stride=2),
#             nn.ReLU(),
#         )
#         with torch.no_grad():
#             dummy = torch.zeros(1, channels * seq_len, num_ranges)
#             conv_out = self.conv(dummy)
#         self.lstm = nn.LSTM(48, 64, bidirectional=True, batch_first=True)
#         self.fc = nn.Linear(64 * 2, hidden_size)

#     def forward(self, observations: torch.Tensor) -> torch.Tensor:
#         b, T, N, C = observations.shape
#         x = observations.permute(0, 3, 1, 2).reshape(b, C * T, N)
#         x = self.conv(x)
#         x = x.permute(0, 2, 1)
#         out, _ = self.lstm(x)
#         out = out.mean(dim=1)
#         return self.fc(out)

# #========================================================
# # Gym environment
# #========================================================
# class LidarSequenceEnv(gym.Env):
#     def __init__(self, sequences, targets):
#         super().__init__()
#         self.sequences = sequences
#         self.targets   = targets
#         self.idx       = 0
#         obs_shape = sequences.shape[1:]
#         self.observation_space = gym.spaces.Box(
#             low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
#         )
#         self.action_space = gym.spaces.Box(
#             low=-1.0, high=1.0, shape=(2,), dtype=np.float32
#         )

#     def reset(self):
#         self.idx = 0
#         return self.sequences[self.idx]

#     def step(self, action):
#         target = self.targets[self.idx]
#         reward = -np.mean((action - target) ** 2)
#         self.idx += 1
#         done = self.idx >= len(self.sequences)
#         obs = self.sequences[self.idx] if not done else np.zeros_like(self.sequences[0])
#         return obs, reward, done, {}

# #========================================================
# # Main
# #========================================================
# if __name__ == '__main__':
#     # Show devices
#     print('TF GPUs:', tf.config.list_physical_devices('GPU'))
#     print('Torch CUDA:', torch.cuda.is_available(), torch.cuda.device_count())

#     # Params
#     npz_paths = ['slow_5min.npz', 'slow_10min.npz']
#     seq_len    = 5
#     batch_size = 64
#     lr         = 5e-5
#     epochs     = 20

#     # Load & preprocess
#     X, y = load_all_data(npz_paths, seq_len)
#     n_samples, _, num_ranges, _ = X.shape
#     X, y = shuffle(X, y, random_state=42)
#     split = int(0.85 * n_samples)
#     X_train, X_test = X[:split], X[split:]
#     y_train, y_test = y[:split], y[split:]

#     # Supervised training
#     tf_model = build_spatiotemporal_model(seq_len, num_ranges)
#     tf_model.compile(optimizer=Adam(lr), loss='huber')
#     history = tf_model.fit(
#         X_train, y_train,
#         validation_data=(X_test, y_test),
#         epochs=epochs,
#         batch_size=batch_size,
#         verbose=2,
#     )
#     plt.plot(history.history['loss'], label='Train')
#     plt.plot(history.history['val_loss'], label='Val')
#     plt.legend(); plt.savefig('Figures/loss_curve.png'); plt.close()

#     # Save TFLite
#     converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
#     converter.target_spec.supported_ops = [
#         tf.lite.OpsSet.TFLITE_BUILTINS,
#         tf.lite.OpsSet.SELECT_TF_OPS
#     ]
#     tflite_model = converter.convert()
#     os.makedirs('Models', exist_ok=True)
#     open('Models/test.tflite','wb').write(tflite_model)

#     # ================= RL² fine-tuning =================
#     env = DummyVecEnv([lambda: LidarSequenceEnv(X_train, y_train)])
#     class RecurrentLidarPolicy(RecurrentActorCriticPolicy):
#         def __init__(self, *args, **kwargs):
#             super().__init__(
#             *args,
#             policy_kwargs=dict(
#                 features_extractor_class=LidarExtractor,
#                 features_extractor_kwargs=dict(hidden_size=64),
#             ),
#             **kwargs)
#             model_rl2 = RecurrentPPO(
#                 policy=RecurrentLidarPolicy,
#                 env=env,
#                 learning_rate=lambda f: f * 3e-5,
#                 n_steps=2048,
#                 batch_size=512,
#                 n_epochs=10,
#                 gamma=0.995,
#                 clip_range=0.2,
#                 ent_coef=5e-4,
#                 tensorboard_log='./rl2_tb/',
#                 device='cuda',
#                 )

            
#             num_trials = 50
#             timesteps_per_trial = 4000  
#             for trial in range(num_trials):
#                 model_rl2.learn(total_timesteps=timesteps_per_trial, reset_num_timesteps=False)
#                 model_rl2.policy.reset_hidden_state()
#                 model_rl2.save(f"Models/ppo_rln_rl2_trial{trial}")