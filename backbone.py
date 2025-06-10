#!/usr/bin/env python3
import os, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle

# ─── Dataset ───
class SequenceDataset(Dataset):
    def __init__(self, npz_paths, seq_len):
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
        X = np.array(X, np.float32)
        y = np.array(y, np.float32)
        X, y = shuffle(X, y, random_state=42)
        self.X, self.y = X, y

    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return torch.from_numpy(self.X[i]), torch.from_numpy(self.y[i])

# ─── Model ───
class AttentionBackbone(nn.Module):
    def __init__(self, seq_len, num_ranges, hidden_size=64, n_heads=4):
        super().__init__()
        C = 2
        # Conv stack
        self.conv = nn.Sequential(
            nn.Conv1d(C*seq_len, 24, 10, stride=4), nn.ReLU(),
            nn.Conv1d(24, 36, 8, stride=4),        nn.ReLU(),
            nn.Conv1d(36, 48, 4, stride=2),        nn.ReLU(),
        )
        # infer L'
        with torch.no_grad():
            dummy = torch.zeros(1, C*seq_len, num_ranges)
            Lp = self.conv(dummy).shape[-1]
        # BiLSTM
        self.lstm = nn.LSTM(48, 64, batch_first=True, bidirectional=True)
        # Attention
        self.attn = nn.MultiheadAttention(128, n_heads, batch_first=True)
        # project to hidden_size
        self.backbone_fc = nn.Linear(128, hidden_size)
        # final head
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # x: [B, seq_len, N, 2]
        B, T, N, C = x.shape
        x = x.permute(0,3,1,2).reshape(B, C*T, N)    # [B, C*T, N]
        x = self.conv(x).permute(0,2,1)            # [B, L', 48]
        out, _ = self.lstm(x)                      # [B, L', 128]
        attn_out, _ = self.attn(out, out, out)     # [B, L', 128]
        ctx = attn_out.mean(dim=1)                 # [B, 128]
        feat = self.backbone_fc(ctx)               # [B, hidden_size]
        return self.head(feat), feat

# ─── Train & Save ───
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',   nargs='+', required=True)
    p.add_argument('--seq_len', type=int, default=5)
    p.add_argument('--epochs',  type=int, default=30)
    p.add_argument('--batch',   type=int, default=64)
    p.add_argument('--lr',      type=float, default=1e-4)
    p.add_argument('--out',     default='Models')
    args = p.parse_args()

    ds = SequenceDataset(args.npz, args.seq_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True)
    _, _, num_ranges, _ = ds.X.shape
    model = AttentionBackbone(args.seq_len, num_ranges)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best=1e9
    os.makedirs(args.out, exist_ok=True)
    for ep in range(args.epochs):
        running=0
        for Xb, yb in dl:
            pred, feat = model(Xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()*Xb.size(0)
        avg = running/len(ds)
        print(f'Epoch {ep+1}: MSE={avg:.4f}')
        if avg<best:
            best=avg
            # snapshot full state
            torch.save({
                'conv': model.conv.state_dict(),
                'lstm': model.lstm.state_dict(),
                'attn': model.attn.state_dict(),
                'backbone_fc': model.backbone_fc.state_dict()
            }, os.path.join(args.out,'backbone_state.pt'))
            # head weights

            W = model.head.weight.detach().cpu()
            b = model.head.bias.detach().cpu()
            torch.save([W, b], "Models/final_head.pt")

    print('Saved backbone_state.pt + final_head.pt in', args.out)

if __name__=='__main__':
    main()
