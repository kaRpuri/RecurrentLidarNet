#!/usr/bin/env python3
import os
import time
import warnings
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import gym

# PyTorch & SB3 imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ROS 2 bag imports
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

#========================================================
# Utility functions
#========================================================
def linear_map(x, x_min, x_max, y_min, y_max):
    return (x - x_min) / (x_max - x_min) * (y_max - y_min) + y_min


def read_ros2_bag(bag_path):
    storage_opts = StorageOptions(uri=bag_path, storage_id='sqlite3')
    conv_opts = ConverterOptions(input_serialization_format='', output_serialization_format='')
    reader = SequentialReader()
    reader.open(storage_opts, conv_opts)

    lidar_data, servo_data, speed_data, timestamps = [], [], [], []
    while reader.has_next():
        topic, serialized_msg, t_ns = reader.read_next()
        t = t_ns * 1e-9
        if topic == 'scan':
            msg = deserialize_message(serialized_msg, LaserScan)
            cleaned = np.nan_to_num(msg.ranges, posinf=0.0, neginf=0.0)
            lidar_data.append(cleaned[::2])
            timestamps.append(t)
        elif topic == 'odom':
            msg = deserialize_message(serialized_msg, Odometry)
            servo_data.append(msg.twist.twist.angular.z)
            speed_data.append(msg.twist.twist.linear.x)
    return np.array(lidar_data), np.array(servo_data), np.array(speed_data), np.array(timestamps)


#========================================================
# Sequence builder
#========================================================
def create_lidar_sequences(lidar_data, servo_data, speed_data, timestamps, sequence_length=5):
    X, y = [], []
    num_ranges = lidar_data.shape[1]
    for i in range(len(lidar_data) - sequence_length):
        frames = np.stack(lidar_data[i: i + sequence_length], axis=0)
        dt = np.diff(timestamps[i: i + sequence_length + 1]).reshape(sequence_length, 1)
        dt_tiled = np.repeat(dt, num_ranges, axis=1)
        seq = np.concatenate([frames[..., None], dt_tiled[..., None]], axis=2)
        X.append(seq)
        y.append([servo_data[i + sequence_length], speed_data[i + sequence_length]])
    return np.array(X), np.array(y)


#========================================================
# Supervised spatiotemporal model (TF)
#========================================================
from tensorflow.keras.layers import Input, TimeDistributed, Conv1D, Flatten, Bidirectional, LSTM, Dense, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_spatiotemporal_model(seq_len, num_ranges):
    inp = Input(shape=(seq_len, num_ranges, 2), name='lidar_sequence')
    x = TimeDistributed(Conv1D(24, 10, strides=4, activation='relu'))(inp)
    x = TimeDistributed(Conv1D(36, 8, strides=4, activation='relu'))(x)
    x = TimeDistributed(Conv1D(48, 4, strides=2, activation='relu'))(x)
    x = TimeDistributed(Flatten())(x)
    lstm_out = Bidirectional(LSTM(64, return_sequences=True))(x)
    q = Dense(64)(lstm_out)
    k = Dense(64)(lstm_out)
    v = Dense(64)(lstm_out)
    attn = Attention()([q, v, k])
    context = tf.reduce_mean(attn, axis=1)
    out = Dense(2, activation='tanh', name='controls')(context)
    return Model(inp, out, name='RNN_Attention_Controller')


#========================================================
# Custom PyTorch feature extractor for SB3
#========================================================
class LidarExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, hidden_size=64):
        super().__init__(observation_space, features_dim=hidden_size)
        seq_len, num_ranges, channels = observation_space.shape
        # conv expects shape [batch, C*T, N]
        self.conv = nn.Sequential(
            nn.Conv1d(channels * seq_len, 24, kernel_size=10, stride=4),
            nn.ReLU(),
            nn.Conv1d(24, 36, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv1d(36, 48, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        # compute output length after conv1d layers manually or via dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, channels * seq_len, num_ranges)
            conv_out = self.conv(dummy)
        conv_len = conv_out.shape[-1]
        self.lstm = nn.LSTM(48, 64, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(64 * 2, hidden_size)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations: [batch, seq_len, num_ranges, 2]
        b, T, N, C = observations.shape
        x = observations.permute(0, 3, 1, 2).reshape(b, C * T, N)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # [b, N', 48]
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        return self.fc(out)


#========================================================
# Gym environment for RL sequences
#========================================================
class LidarSequenceEnv(gym.Env):
    def __init__(self, sequences, targets):
        super().__init__()
        self.sequences = sequences
        self.targets = targets
        self.idx = 0
        obs_shape = sequences.shape[1:]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self):
        self.idx = 0
        return self.sequences[self.idx]

    def step(self, action):
        target = self.targets[self.idx]
        reward = -np.mean((action - target) ** 2)
        self.idx += 1
        done = self.idx >= len(self.sequences)
        obs = self.sequences[self.idx] if not done else np.zeros_like(self.sequences[0])
        return obs, reward, done, {}


#========================================================
# Main execution
#========================================================
if __name__ == '__main__':
    print('GPU AVAILABLE:', torch.cuda.is_available())

    # Parameters
    bag_paths = [
        './car_Dataset/controller_slow_5min/controller_slow_5min_0.db3',
        './car_Dataset/controller_slow_10min/controller_slow_10.db3',
    ]
    seq_len = 5
    batch_size = 64
    lr = 5e-5
    epochs = 20

    # Load and concatenate bags
    all_lidar, all_servo, all_speed, all_ts = [], [], [], []
    for pth in bag_paths:
        l, s, sp, ts = read_ros2_bag(pth)
        print(f'Loaded {len(l)} scans from {pth}')
        all_lidar.extend(l)
        all_servo.extend(s)
        all_speed.extend(sp)
        all_ts.extend(ts)
    all_lidar = np.array(all_lidar)
    all_servo = np.array(all_servo)
    all_speed = np.array(all_speed)
    all_ts = np.array(all_ts)

    # Normalize speed 0→1
    min_s, max_s = all_speed.min(), all_speed.max()
    all_speed = linear_map(all_speed, min_s, max_s, 0, 1)

    # Build sequences
    X, y = create_lidar_sequences(all_lidar, all_servo, all_speed, all_ts, seq_len)
    n_samples, _, num_ranges, _ = X.shape
    print(f'Total sequences: {n_samples}, ranges per scan: {num_ranges}')

    # Shuffle & split
    X, y = shuffle(X, y, random_state=42)
    split = int(0.85 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Supervised training (TF)
    model_tf = build_spatiotemporal_model(seq_len, num_ranges)
    model_tf.compile(optimizer=Adam(lr), loss='huber')
    print(model_tf.summary())
    t0 = time.time()
    history = model_tf.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
    )
    print(f'Training done in {int(time.time() - t0)}s')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('Figures/loss_curve.png'); plt.close()
    converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    os.makedirs('Models', exist_ok=True)
    with open('Models/test.tflite', 'wb') as f:
        f.write(tflite_model)
    print('TFLite model saved.')
    test_loss = model_tf.evaluate(X_test, y_test, verbose=0)
    print(f'Final test loss: {test_loss:.4f}')

    # RL fine-tuning (PyTorch + SB3)
    env = DummyVecEnv([lambda: LidarSequenceEnv(X_train, y_train)])
    policy_kwargs = dict(
        features_extractor_class=LidarExtractor,
        features_extractor_kwargs=dict(hidden_size=64),
    )
    model_rl = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=lr,
        n_steps=seq_len * 20,
        batch_size=batch_size,
        gamma=0.99,
        verbose=1,
        tensorboard_log='./rl2_tb/',
        policy_kwargs=policy_kwargs,
    )
    model_rl.learn(total_timesteps=100_000)
    os.makedirs('Models', exist_ok=True)
    model_rl.save('Models/ppo_rln_sb3')
