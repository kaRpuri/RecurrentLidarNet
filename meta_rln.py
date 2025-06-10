#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import gym
from torch.utils.data import Dataset
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# ─── Data loader ───
def load_npz(npz_paths, seq_len):
    all_l, all_s, all_sp, all_t = [], [], [], []
    for p in npz_paths:
        d = np.load(p)
        all_l.append(d['lidar']);   all_s.append(d['servo'])
        all_sp.append(d['speed']);  all_t.append(d['ts'])
    L = np.concatenate(all_l, axis=0)
    S = np.concatenate(all_s, axis=0)
    P = np.concatenate(all_sp, axis=0)
    T = np.concatenate(all_t, axis=0)
    X, y = [], []
    N = L.shape[1]
    for i in range(len(L) - seq_len):
        frames = L[i:i+seq_len]
        dt = np.diff(T[i:i+seq_len+1]).reshape(seq_len,1)
        dt = np.repeat(dt, N, axis=1)
        X.append(np.stack([frames, dt], axis=2))
        y.append([S[i+seq_len], P[i+seq_len]])
    return np.array(X, np.float32), np.array(y, np.float32)

# ─── Env definitions ───
class LidarSequenceEnv(gym.Env):
    def __init__(self, X, y):
        super().__init__()
        self.X, self.y = X, y
        self.idx = 0
        seq_len, num_ranges, _ = X.shape[1:]
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
            shape=(seq_len, num_ranges, 2), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0,
            shape=(2,), dtype=np.float32)
    def reset(self):
        self.idx = 0
        return self.X[0]
    def step(self, action):
        target = self.y[self.idx]
        reward = -np.mean((action - target)**2)
        self.idx += 1
        done = self.idx >= len(self.X)
        obs = self.X[self.idx] if not done else np.zeros_like(self.X[0])
        return obs, reward, done, {}

class MetaDriveEnv(gym.Env):
    def __init__(self, segments):
        super().__init__()
        self.segments, self.task_idx = segments, 0
        self._swap()
    def _swap(self):
        X, y = self.segments[self.task_idx]
        self.env = LidarSequenceEnv(X, y)
        self.task_idx = (self.task_idx + 1) % len(self.segments)
        self.observation_space = self.env.observation_space
        self.action_space      = self.env.action_space
    def reset(self):
        self._swap()
        return self.env.reset()
    def step(self, action):
        return self.env.step(action)

# ─── Sup-trained backbone (PyTorch) ───
class AttentionBackbone(nn.Module):
    def __init__(self, seq_len, num_ranges, hidden_size=64, n_heads=4):
        super().__init__()
        C = 2
        self.conv = nn.Sequential(
            nn.Conv1d(C*seq_len, 24, 10, stride=4), nn.ReLU(),
            nn.Conv1d(24, 36, 8, stride=4),       nn.ReLU(),
            nn.Conv1d(36, 48, 4, stride=2),       nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, C*seq_len, num_ranges)
            _ = self.conv(dummy)
        self.lstm = nn.LSTM(48, 64, batch_first=True, bidirectional=True)
        self.attn = nn.MultiheadAttention(128, n_heads, batch_first=True)
        self.backbone_fc = nn.Linear(128, hidden_size)
        self.head        = nn.Linear(hidden_size, 2)
    def forward(self, x):
        B,T,N,C = x.shape
        x = x.permute(0,3,1,2).reshape(B, C*T, N)
        x = self.conv(x).permute(0,2,1)
        out,_ = self.lstm(x)
        attn_out,_ = self.attn(out,out,out)
        ctx = attn_out.mean(dim=1)
        feat = self.backbone_fc(ctx)
        return self.head(feat), feat

# ─── SB3 FeaturesExtractor ───
class AttentionExtractor(BaseFeaturesExtractor):
    def __init__(self, obs_space, hidden_size=64, n_heads=4):
        super().__init__(obs_space, features_dim=hidden_size)
        seq_len, num_ranges, _ = obs_space.shape
        bb = AttentionBackbone(seq_len, num_ranges, hidden_size, n_heads)
        payload = torch.load("Models/backbone_state.pt")
        bb.conv.load_state_dict(payload['conv'])
        bb.lstm.load_state_dict(payload['lstm'])
        bb.attn.load_state_dict(payload['attn'])
        bb.backbone_fc.load_state_dict(payload['backbone_fc'])
        self.conv, self.lstm, self.attn, self.fc = \
            bb.conv, bb.lstm, bb.attn, bb.backbone_fc

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        b,T,N,C = obs.shape
        x = obs.permute(0,3,1,2).reshape(b, C*T, N)
        x = self.conv(x).permute(0,2,1)
        out,_ = self.lstm(x)
        attn_out,_ = self.attn(out,out,out)
        ctx = attn_out.mean(dim=1)
        return self.fc(ctx)

# ─── NaN checker ───
class NanDetectionCallback(BaseCallback):
    def _on_rollout_start(self):
        if np.isnan(self.model.rollout_buffer.observations).any():
            raise RuntimeError("NaN in rollout buffer")
    def _on_step(self): return True

class EarlyStopOnValMSE(BaseCallback):
    def __init__(self, val_X, val_y, eval_env, eval_freq, patience, verbose=1):
        super().__init__(verbose)
        self.val_X, self.val_y = val_X, val_y
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_mse = float('inf')
        self.no_improve = 0
        self.n_calls = 0

    def _on_step(self) -> bool:
        self.n_calls += 1
        if self.n_calls % self.eval_freq != 0:
            return True

        # compute MSE on val set
        preds = []
        for seq in self.val_X:
            action, _ = self.model.predict(seq[None], deterministic=True)
            preds.append(action[0])
        preds = np.stack(preds)
        mse = ((preds - self.val_y)**2).mean()

        if self.verbose:
            print(f"[Eval] Step {self.num_timesteps}: val MSE = {mse:.5f} (best {self.best_mse:.5f})")

        if mse + 1e-6 < self.best_mse:
            self.best_mse = mse
            self.no_improve = 0
        else:
            self.no_improve += 1

        if self.no_improve >= self.patience:
            print(f"Stopping early: no val‐MSE improvement in {self.patience} evals.")
            return False

        return True

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--npz',       nargs='+', required=True)
    p.add_argument('--device',    choices=['cpu','cuda','mps'], default='cpu')
    p.add_argument('--meta_steps',type=int, default=500_000)
    p.add_argument('--n_steps',   type=int, default=50)
    p.add_argument('--eval_freq', type=int, default=50_000)
    p.add_argument('--patience',  type=int, default=3)
    args = p.parse_args()

    # load & split into train/val
    X, y = load_npz(args.npz, seq_len=5)
    split = int(0.85 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_val,   y_val   = X[split:], y[split:]

    # create meta‐train env (only using train segments)
    train_segs = list(zip(np.array_split(X_train,20), np.array_split(y_train,20)))
    train_env = VecNormalize(DummyVecEnv([lambda: MetaDriveEnv(train_segs)]),
                             norm_obs=True, norm_reward=False, clip_obs=10.0)

    # create a simple eval_env for val‐MSE (deterministic, no VecNormalize)
    val_env = DummyVecEnv([lambda: MetaDriveEnv([(X_val, y_val)])])

    # linear LR schedule from 1e-5 down to 1e-6
    initial_lr = 1e-5
    def lr_schedule(progress_remaining):
        return 1e-6 + progress_remaining * (initial_lr - 1e-6)

    policy_kwargs = dict(
        features_extractor_class=AttentionExtractor,
        features_extractor_kwargs=dict(hidden_size=64, n_heads=4),
        net_arch=dict(pi=[], vf=[]),
    )

    model = PPO(
        ActorCriticPolicy,
        train_env,
        n_steps=args.n_steps,
        batch_size=args.n_steps * 2,       # larger batch
        learning_rate=lr_schedule,         # decaying LR
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="meta_rl_tb/",
        policy_kwargs=policy_kwargs,
        device=args.device
    )

    # head injection (as before)...
    W_loaded, b_loaded = torch.load("Models/final_head.pt")
    W = torch.tensor(W_loaded, device=model.device)
    b = torch.tensor(b_loaded, device=model.device)
    head = model.policy.action_net
    if isinstance(head, nn.Sequential):
        head = head[0]
    out_dim, in_dim = head.weight.shape
    if W.shape == (in_dim, out_dim):
        W = W.T
    elif W.shape != (out_dim, in_dim):
        raise ValueError("Head shape mismatch")
    with torch.no_grad():
        head.weight.copy_(W); head.bias.copy_(b)

    for name, param in model.policy.named_parameters():
        param.requires_grad = name.startswith("action_net.")

    # callbacks: early stop + NaN check
    early_stop_cb = EarlyStopOnValMSE(
        X_val, y_val, val_env,
        eval_freq=args.eval_freq,
        patience=args.patience
    )
    nan_cb = NanDetectionCallback()

    model.learn(total_timesteps=args.meta_steps,
                callback=[nan_cb, early_stop_cb])

    os.makedirs("Models", exist_ok=True)
    model.save("Models/meta_rl_finetuned_head.zip")
    print("🎉 Saved fine-tuned head model to Models/meta_rl_finetuned_head.zip")