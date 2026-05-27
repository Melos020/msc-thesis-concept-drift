"""Training loops (thesis Ch 4.3).

TCN: BCE loss, Adam lr=1e-3, batch 64, early stop patience 10 on 20% val split.
ANFIS: closed-form ridge-LSE consequent fit after k-means premise init.

Both produce a `predict_proba` method returning P(up) on a windowed sequence.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple

from ..models.tcn import TCN
from ..models.anfis import ANFIS


def make_windows(X: np.ndarray, y: np.ndarray, seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sliding windows of length seq_len. Returns (Xw, yw)."""
    if len(X) <= seq_len:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    Xw = np.stack([X[i:i+seq_len] for i in range(len(X) - seq_len)], axis=0)
    yw = y[seq_len:]
    return Xw.astype(np.float32), yw.astype(np.float32)


def train_tcn(X_train: np.ndarray, y_train: np.ndarray,
              seq_len: int = 30, epochs: int = 50,
              batch_size: int = 64, lr: float = 1e-3,
              patience: int = 10, seed: int = 0,
              device: str = 'cpu') -> Tuple[TCN, StandardScaler]:
    """Train a TCN with early stopping on 20% val split."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train).astype(np.float32)
    
    Xw, yw = make_windows(X_scaled, y_train, seq_len)
    if len(Xw) == 0:
        m = TCN(input_dim=X_train.shape[1]).to(device)
        return m, scaler
    
    # 20% val split (last 20% to preserve temporal order)
    split = int(0.8 * len(Xw))
    Xtr, ytr = Xw[:split], yw[:split]
    Xva, yva = Xw[split:], yw[split:]
    
    model = TCN(input_dim=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    
    tr_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                           batch_size=batch_size, shuffle=True)
    Xva_t = torch.tensor(Xva).to(device)
    yva_t = torch.tensor(yva).to(device)
    
    best_val = float('inf')
    best_state = None
    no_improve = 0
    
    for ep in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = bce(logits, yb)
            loss.backward()
            opt.step()
        # Val
        model.eval()
        with torch.no_grad():
            if len(Xva_t) > 0:
                v_loss = bce(model(Xva_t), yva_t).item()
            else:
                v_loss = 0.0
        if v_loss < best_val - 1e-5:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, scaler


def predict_tcn(model: TCN, scaler: StandardScaler, X: np.ndarray,
                seq_len: int = 30, device: str = 'cpu') -> np.ndarray:
    """Return P(up) for each timestep after warm-up."""
    X_scaled = scaler.transform(X).astype(np.float32)
    Xw, _ = make_windows(X_scaled, np.zeros(len(X_scaled)), seq_len)
    if len(Xw) == 0:
        return np.zeros((0,))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(Xw).to(device))
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs


def train_anfis(X_train: np.ndarray, y_train: np.ndarray,
                n_mfs: int = 3, seed: int = 0) -> Tuple[ANFIS, StandardScaler]:
    """Fit ANFIS premise (k-means, seeded) and consequent (ridge LSE)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train).astype(np.float32)
    model = ANFIS(input_dim=X_train.shape[1], n_mfs=n_mfs)
    model.fit_premise(Xs, seed=seed)
    model.fit_consequent(Xs, y_train.astype(np.float32))
    return model, scaler


def predict_anfis(model: ANFIS, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Return P(up) for each row (no windowing — ANFIS is point-wise)."""
    Xs = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(Xs))
        probs = torch.sigmoid(logits).numpy()
    return probs
