#!/usr/bin/env python3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from stable_baselines3 import PPO
import gym

# 1) load your test sequences (same preprocessing you used in training)
def load_npz(npz_paths, seq_len):
    all_l, all_s, all_sp, all_t = [], [], [], []
    for p in npz_paths:
        d = np.load(p)
        all_l.append(d['lidar']); all_s.append(d['servo'])
        all_sp.append(d['speed']); all_t.append(d['ts'])
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
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# 2) load base TFLite model
tflite_path = "Models/RNN_Attn_Controller.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_path)
interpreter.allocate_tensors()
inp_detail = interpreter.get_input_details()[0]
out_detail = interpreter.get_output_details()[0]

# 3) load supervised Keras controller & RL fine-tuned policy
sup_model = load_model("Models/sup_controller.h5", compile=False)

# load your fine-tuned head PPO policy
META_RL_PATH = "Models/meta_rl_finetuned_head.zip"
meta_rl = PPO.load(META_RL_PATH)

# 4) run inference on test set
seq_len = inp_detail["shape"][1]
X, y = load_npz(["slow_5min.npz","slow_10min.npz"], seq_len)
split = int(0.85*len(X))
X_test, y_test = X[split:], y[split:]

base_preds, tuned_preds = [], []
for seq, tgt in zip(X_test, y_test):
    inp = seq[None,...]

    # a) base TFLite
    interpreter.set_tensor(inp_detail['index'], inp)
    interpreter.invoke()
    base_preds.append(interpreter.get_tensor(out_detail['index'])[0])

    # b) meta-RL tuned via PPO
    action, _ = meta_rl.predict(inp, deterministic=True)
    tuned_preds.append(action[0] if action.ndim>1 else action)

base_preds   = np.stack(base_preds)
tuned_preds  = np.stack(tuned_preds)
errors_base  = np.mean((base_preds  - y_test)**2, axis=1)
errors_tuned = np.mean((tuned_preds - y_test)**2, axis=1)

print(f"Base TFLite   MSE: {errors_base.mean():.4f} ± {errors_base.std():.4f}")
print(f"Meta-RL PPO   MSE: {errors_tuned.mean():.4f} ± {errors_tuned.std():.4f}")
