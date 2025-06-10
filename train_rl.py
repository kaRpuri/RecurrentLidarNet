#!/usr/bin/env python3
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ─── Data + Sequences ───
def create_lidar_sequences(lidar, servo, speed, ts, seq_len=5):
    X, y = [], []
    N = lidar.shape[1]
    for i in range(len(lidar) - seq_len):
        frames = lidar[i:i+seq_len]
        dt = np.diff(ts[i:i+seq_len+1]).reshape(seq_len,1)
        dt = np.repeat(dt, N, axis=1)
        X.append(np.stack([frames, dt], axis=2))
        y.append([servo[i+seq_len], speed[i+seq_len]])
    return np.array(X, np.float32), np.array(y, np.float32)

def load_all(npz_paths, seq_len):
    all_l, all_s, all_sp, all_t = [], [], [], []
    for p in npz_paths:
        d = np.load(p)
        all_l.append(d['lidar'])
        all_s.append(d['servo'])
        all_sp.append(d['speed'])
        all_t.append(d['ts'])
    L = np.concatenate(all_l, axis=0)
    S = np.concatenate(all_s, axis=0)
    P = np.concatenate(all_sp, axis=0)
    T = np.concatenate(all_t, axis=0)
    return create_lidar_sequences(L, S, P, T, seq_len)

# ─── Supervised RNN Controller ───
def build_controller(seq_len, num_ranges):
    inp = Input((seq_len, num_ranges, 2))
    x = TimeDistributed(Conv1D(24,10,4,activation='relu'))(inp)
    x = TimeDistributed(Conv1D(36,8,4,activation='relu'))(x)
    x = TimeDistributed(Conv1D(48,4,2,activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    out = Dense(2, activation='tanh', name='final_dense')(x)
    return Model(inp, out, name='controller')

# ─── Environment that only perturbs final Dense ───
class RLTunerEnv(gym.Env):
    def __init__(self, X, y, seq_len, num_ranges, sup_model_path):
        super().__init__()
        self.X, self.y = X, y
        self.idx = 0

        # load the full controller
        self.controller = tf.keras.models.load_model(sup_model_path)
        # grab original final‐dense weights
        self.final_layer = self.controller.get_layer('final_dense')
        W, b = self.final_layer.get_weights()
        self.orig_W, self.orig_b = W.copy(), b.copy()
        # action dim = W.size + b.size
        self.action_dim = W.size + b.size

        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf,
            shape=(seq_len, num_ranges, 2),
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1e-2, high=1e-2,
            shape=(self.action_dim,),
            dtype=np.float32
        )

    def reset(self):
        self.idx = 0
        return self.X[0]

    def step(self, delta):
        # split into W and b deltas
        dW = delta[:self.orig_W.size].reshape(self.orig_W.shape)
        db = delta[self.orig_W.size:].reshape(self.orig_b.shape)
        # apply
        self.final_layer.set_weights([self.orig_W + dW, self.orig_b + db])

        # compute reward = –MSE
        inp = self.X[self.idx:self.idx+1]
        pred = self.controller.predict(inp, verbose=0)[0]
        target = self.y[self.idx]
        loss = np.mean((pred - target)**2)
        reward = -float(loss)

        # restore
        self.final_layer.set_weights([self.orig_W, self.orig_b])

        # advance
        self.idx += 1
        done = self.idx >= len(self.X)
        obs = self.X[self.idx] if not done else np.zeros_like(self.X[0])
        return obs, reward, done, {}

# ─── Main ───
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',          nargs='+', required=True)
    p.add_argument('--sup_epochs',   type=int, default=50)
    p.add_argument('--rl_timesteps', type=int, default=200_000)
    p.add_argument('--sup_model',    default='Models/sup_controller.keras')
    p.add_argument('--rl_model',     default='Models/ppo_rltuner.zip')
    opts = p.parse_args()

    seq_len = 5
    X, y = load_all(opts.npz, seq_len)
    N = X.shape[2]

    # 1) Supervised train
    model = build_controller(seq_len, N)
    model.compile(Adam(1e-4), loss='mse')
    es = EarlyStopping('val_loss', patience=5, restore_best_weights=True)
    model.fit(X, y, validation_split=0.15,
              epochs=opts.sup_epochs, batch_size=64,
              callbacks=[es], verbose=2)

    os.makedirs(os.path.dirname(opts.sup_model), exist_ok=True)
    model.save(opts.sup_model)
    print("Saved supervised model →", opts.sup_model)

    # 2) RL fines‐tuning only final‐dense
    env = DummyVecEnv([lambda: RLTunerEnv(X, y, seq_len, N, opts.sup_model)])
    rl = PPO('MlpPolicy', env,
             learning_rate=1e-5,
             n_steps=64, batch_size=16,
             n_epochs=5,
             verbose=1,
             tensorboard_log='./rltuner_tb/')
    rl.learn(total_timesteps=opts.rl_timesteps)
    os.makedirs(os.path.dirname(opts.rl_model), exist_ok=True)
    rl.save(opts.rl_model)
    print("Saved RL fine-tuned policy →", opts.rl_model)

if __name__ == '__main__':
    main()
