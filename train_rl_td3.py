#!/usr/bin/env python3
import os, argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.utils import shuffle

import gym
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

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


class SmoothRLTunerEnv(gym.Env):
    def __init__(self, X, y, seq_len, num_ranges, sup_model_path, alpha=0.5):
        super().__init__()
        self.X, self.y = X, y
        self.seq_len = seq_len
        self.num_ranges = num_ranges
        self.alpha = alpha
        self.idx = 0

        # load the full controller and grab its final Dense layer
        self.controller = load_model(sup_model_path)
        final = self.controller.get_layer('final_dense')
        W, b = final.get_weights()
        self.final_layer = final
        self.orig_W, self.orig_b = W.copy(), b.copy()

        # action space is the flattened perturbation of W and b
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

        # buffer to compute jerk (change in the weight-delta vector)
        self.prev_delta = np.zeros(self.action_dim, dtype=np.float32)

    def reset(self):
        self.idx = 0
        self.prev_delta.fill(0.0)
        return self.X[0]

    def step(self, delta):
        # split delta into weight‐ and bias‐perturbations
        dW = delta[:self.orig_W.size].reshape(self.orig_W.shape)
        db = delta[self.orig_W.size:].reshape(self.orig_b.shape)

        # apply the perturbation
        self.final_layer.set_weights([self.orig_W + dW, self.orig_b + db])

        # forward pass & compute MSE loss
        inp = self.X[self.idx:self.idx+1]
        pred = self.controller.predict(inp, verbose=0)[0]
        target = self.y[self.idx]
        loss = float(np.mean((pred - target)**2))

        # jerk penalty on the full weight‐delta vector
        jerk = float(np.mean(np.abs(delta - self.prev_delta)))
        reward = -loss - self.alpha * jerk

        # restore original weights
        self.final_layer.set_weights([self.orig_W, self.orig_b])

        # advance time-step
        self.prev_delta[:] = delta
        self.idx += 1
        done = self.idx >= len(self.X)
        obs = self.X[self.idx] if not done else np.zeros_like(self.X[0])

        return obs, reward, done, {}


# ─── Main ───
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',          nargs='+', required=True)
    p.add_argument('--sup_model',    required=True)
    p.add_argument('--output_td3',   default='Models/td3_rltuner')
    p.add_argument('--timesteps',    type=int, default=200_000)
    p.add_argument('--alpha',        type=float, default=0.5,
                   help="jerk penalty weight")
    p.add_argument('--seq_len',      type=int, default=5)
    opts = p.parse_args()

    # load & split
    X, y = load_all(opts.npz, opts.seq_len)
    X, y = shuffle(X, y, random_state=42)
    seq_len, num_ranges, _ = X.shape[1:]

    # make normalizing VecEnv
    base_env = DummyVecEnv([lambda: SmoothRLTunerEnv(
        X, y, opts.seq_len, num_ranges, opts.sup_model, alpha=opts.alpha
    )])
    env = VecNormalize(base_env, norm_obs=True, norm_reward=True, clip_reward=10.0)

    # action noise
    n_act = env.action_space.shape[-1]
    noise = NormalActionNoise(
        mean=np.zeros(n_act), sigma=0.05 * np.ones(n_act)
    )

    # train TD3
    model = TD3(
        "MlpPolicy", env,
        action_noise=noise,
        learning_rate=1e-4,
        batch_size=256,
        buffer_size=100_000, 
        verbose=1,
        tensorboard_log="./td3_rltuner_tb/",
        device="cuda",
        policy_kwargs=dict(net_arch=[256,256])
    )
    model.learn(total_timesteps=opts.timesteps)
    os.makedirs(os.path.dirname(opts.output_td3), exist_ok=True)
    model.save(opts.output_td3)
    print("Saved TD3-tuned policy →", opts.output_td3)

if __name__ == '__main__':
    main()
