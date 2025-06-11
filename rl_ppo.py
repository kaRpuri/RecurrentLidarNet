import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv1D, Flatten,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import torch
from stable_baselines3 import PPO, DDPG, TD3, SAC
from sb3_contrib import RecurrentPPO
from absl import logging
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, unwrap_vec_normalize
from stable_baselines3.common.monitor import Monitor
from functools import partial
from tqdm import tqdm
import gym
from gym import spaces
from rl_env import F110GymWrapper
from stablebaseline3.feature_extractor import (
    F1TenthFeaturesExtractor, MLPFeaturesExtractor,
    ResNetFeaturesExtractor, TransformerFeaturesExtractor,
    MoEFeaturesExtractor
)
from imitation.algorithms import bc
from imitation.data import types as data_types
from imitation.data.rollout import flatten_trajectories
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sortedcontainers import SortedList
import datetime
import time

# Configure TensorFlow to use GPU (e.g. GTX 4070)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logging.info(f"TensorFlow GPUs: {[gpu.name for gpu in gpus]}")

# ─── Data + Sequences ──────────────────────────────────────────────────────────
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

# ─── Supervised Controller ─────────────────────────────────────────────────────
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

# ─── Feature extractor selector ───────────────────────────────────────────────
def get_feature_extractor_class(name):
    return {
        'FILM': F1TenthFeaturesExtractor,
        'MLP': MLPFeaturesExtractor,
        'RESNET': ResNetFeaturesExtractor,
        'TRANSFORMER': TransformerFeaturesExtractor,
        'MOE': MoEFeaturesExtractor
    }.get(name, F1TenthFeaturesExtractor)

# ─── Create PPO ───────────────────────────────────────────────────────────────
def create_ppo(env, seed, include_params, feat_name, lidar_mode):
    lidar_dim = {'NONE':0,'FULL':1080,'DOWNSAMPLED':108}[lidar_mode]
    feat_cls = get_feature_extractor_class(feat_name)
    policy_kwargs = {
        'features_extractor_class': feat_cls,
        'features_extractor_kwargs':{
            'features_dim':256,'state_dim':4,
            'lidar_dim':lidar_dim,'param_dim':12,
            'include_params':include_params,
            'include_lidar':lidar_mode!='NONE'
        },
        'net_arch':[dict(pi=[256,128,64],vf=[256,128,64])]
    }
    return PPO(
        'MlpPolicy', env,
        learning_rate=3e-4,
        n_steps=2048, batch_size=512,
        n_epochs=10, gamma=0.99,
        gae_lambda=0.95, clip_range=0.2,
        ent_coef=0.01, vf_coef=1,
        tensorboard_log='./ppo_tb/', seed=seed,
        policy_kwargs=policy_kwargs,
        device='cuda', verbose=1
    )

# ─── Reward callback ──────────────────────────────────────────────────────────
class RewardCallback(BaseCallback):
    def __init__(self, window=10, verbose=1):
        super().__init__(verbose)
        self.window = window
        self.ep_returns = []
    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                r = info['episode']['r']
                self.ep_returns.append(r)
                if len(self.ep_returns)>=self.window:
                    avg = np.mean(self.ep_returns[-self.window:])
                    print(f"[Episode {len(self.ep_returns)}] Avg reward: {avg:.2f}")
        return True

# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--npz', nargs='+', required=True)
    p.add_argument('--sup_epochs',type=int,default=50)
    p.add_argument('--rl_timesteps',type=int,default=200_000)
    p.add_argument('--sup_model',default='Models/sup_controller.keras')
    p.add_argument('--rl_model',default='Models/ppo_rltuner.zip')
    p.add_argument('--feature',default='FILM')
    p.add_argument('--include_params',action='store_true')
    p.add_argument('--lidar_mode',choices=['NONE','FULL','DOWNSAMPLED'],default='FULL')
    args=p.parse_args()

    # supervised
    X,y=load_all(args.npz,5)
    N=X.shape[2]
    sup=build_controller(5,N)
    sup.compile(Adam(1e-4),'mse')
    sup.fit(X,y,validation_split=0.15,epochs=args.sup_epochs,
            batch_size=64,callbacks=[EarlyStopping('val_loss',patience=5,restore_best_weights=True)],verbose=2)
    os.makedirs(os.path.dirname(args.sup_model),exist_ok=True)
    sup.save(args.sup_model)
    print(f"Saved supervised → {args.sup_model}")

    # RL
    def mk(): return Monitor(RLTunerEnv(X,y,5,N,args.sup_model))
    env=DummyVecEnv([mk]); env=VecNormalize(env,norm_obs=True,norm_reward=True,clip_obs=10.)
    model=create_ppo(env,seed=0,include_params=args.include_params,
                     feat_name=args.feature,lidar_mode=args.lidar_mode)
    cb=RewardCallback(window=10)
    model.learn(total_timesteps=args.rl_timesteps,callback=cb)
    os.makedirs(os.path.dirname(args.rl_model),exist_ok=True)
    model.save(args.rl_model)
    print(f"Saved RL → {args.rl_model}")
