"""
run_enriched.py  --  self-contained TCN trainer for the hourly pipeline.

Provides the three functions run_hourly.py imports:
    train_tcn_strong(Xtr, ytr, seq_len, seed)  -> (model, scaler)
    predict_tcn_strong(model, scaler, X, seq_len) -> probability array
    make_windows_seq(X, seq_len)               -> (B, seq_len, F) sequences

This mirrors the thesis Ch 4.3 configuration: the shipped 3-block TCN
(dilations 1/2/4, 64 channels, kernel 3, dropout 0.15), trained with Adam,
BCEWithLogitsLoss, early stopping on a held-in tail of the training segment.
All preprocessing (StandardScaler) is fit on training data only, so the
pipeline stays leakage-free. Sequences are built causally: row t is predicted
from the seq_len bars ending at t.

If you already have your own run_enriched.py from the original experiments,
delete this file and use yours -- the interface is identical.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.models.tcn import TCN

torch.set_num_threads(4)


def make_windows_seq(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Build causal sequences aligned to rows [seq_len:].
    Output[i] corresponds to X row (i + seq_len) and contains the seq_len bars
    ending at that row: X[i+1 : i+seq_len+1]. Length = len(X) - seq_len, which
    matches how eval_window slices labels (ypre[seq_len:], ypost[seq_len:])."""
    X = np.asarray(X, dtype=np.float32)
    n, f = X.shape
    if n <= seq_len:
        return np.zeros((0, seq_len, f), dtype=np.float32)
    m = n - seq_len
    out = np.zeros((m, seq_len, f), dtype=np.float32)
    for i in range(m):
        out[i] = X[i + 1: i + seq_len + 1]
    return out


def train_tcn_strong(Xtr, ytr, seq_len=24, seed=0,
                     epochs=60, lr=1e-3, patience=8, val_frac=0.15,
                     channels=64, dropout=0.15):
    """Train the thesis TCN on one window's training segment.
    Returns (model, scaler). Deterministic given seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    Xtr = np.asarray(Xtr, dtype=np.float32)
    ytr = np.asarray(ytr, dtype=np.float32)

    scaler = StandardScaler().fit(Xtr)
    Xs = scaler.transform(Xtr).astype(np.float32)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)

    seqs = make_windows_seq(Xs, seq_len)               
    ylab = ytr[seq_len:]                                
    if len(seqs) < 32:
        class _Const:
            def __init__(self, p): self.p = float(p)
        return _Const(ytr.mean()), scaler

    n = len(seqs)
    n_val = max(int(round(n * val_frac)), 16)
    tr_idx = slice(0, n - n_val)
    va_idx = slice(n - n_val, n)

    Xtr_t = torch.from_numpy(seqs[tr_idx])
    ytr_t = torch.from_numpy(ylab[tr_idx].astype(np.float32))
    Xva_t = torch.from_numpy(seqs[va_idx])
    yva_t = torch.from_numpy(ylab[va_idx].astype(np.float32))

    model = TCN(input_dim=Xs.shape[1], channels=channels, dropout=dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # class-imbalance-aware positive weight
    pos = float(ytr_t.sum()); neg = float(len(ytr_t) - pos)
    pw = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_val = float("inf"); best_state = None; bad = 0
    bs = 256
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(Xtr_t), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            logit = model(Xtr_t[j])
            loss = loss_fn(logit, ytr_t[j])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vlogit = model(Xva_t)
            vloss = float(loss_fn(vlogit, yva_t))
        if vloss < best_val - 1e-4:
            best_val = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, scaler


def predict_tcn_strong(model, scaler, X, seq_len=24):
    """Probability of the positive class for rows [seq_len:] of X (aligned to
    make_windows_seq output, matching eval_window's label slicing)."""
    if model.__class__.__name__ == "_Const":
        m = max(len(X) - seq_len, 0)
        return np.full(m, float(getattr(model, "p", 0.5)))
    X = np.asarray(X, dtype=np.float32)
    Xs = scaler.transform(X).astype(np.float32)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
    seqs = make_windows_seq(Xs, seq_len)
    if len(seqs) == 0:
        return np.array([], dtype=float)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(seqs))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs.astype(float)
