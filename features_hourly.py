"""Hourly feature engine + targets for the TCN-vs-ANFIS drift study.

Feature design is lifted from the user's own production quant system
(tradebot wave-13 CORE_FEATURES + microstructure), adapted to pure OHLCV
(funding / OI / liquidation columns are unavailable in the public klines, so
those are dropped; everything else is reproduced). All features are causal.

Targets:
  vol_regime_h12   PRIMARY/predictable: will next-12h realized vol be in the
                   upper tercile of its trailing distribution? (vol clustering
                   -> genuinely learnable -> drift has competence to disrupt)
  large_move_h12   will |next-12h return| exceed trailing 70th pct? (magnitude)
  tb_dir_h12       triple-barrier direction (ATR-scaled, after-cost) -- the
                   user's own label; kept as the "hard" directional target
"""
from __future__ import annotations
import numpy as np, pandas as pd


def load_klines(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    return d[["open", "high", "low", "close", "volume"]].astype(float)


def _rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    return (100 - 100 / (1 + up / dn.replace(0, np.nan))) / 100.0


def _atr(h, l, c, n=14):
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def _adx(h, l, c, n=14):
    up = h.diff(); dn = -l.diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr, index=c.index).rolling(n).mean()
    pdi = 100 * pd.Series(plus, index=c.index).rolling(n).mean() / atr.replace(0, np.nan)
    mdi = 100 * pd.Series(minus, index=c.index).rolling(n).mean() / atr.replace(0, np.nan)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return (dx.rolling(n).mean() / 100.0)


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = d.open, d.high, d.low, d.close, d.volume
    lr = np.log(c / c.shift(1))
    f = pd.DataFrame(index=d.index)
    # CORE (OHLCV-derivable subset of the user's CORE_FEATURES) 
    f["ret_1"] = lr
    f["ret_3"] = np.log(c / c.shift(3))
    f["ret_6"] = np.log(c / c.shift(6))
    f["ret_12"] = np.log(c / c.shift(12))
    atr14 = _atr(h, l, c, 14)
    f["atr_14"] = atr14 / c
    f["atr_pctile_30d"] = (atr14 / c).rolling(24 * 30).rank(pct=True)
    f["rsi_14"] = _rsi(c, 14)
    f["adx_14"] = _adx(h, l, c, 14)
    f["volume_zscore_24"] = (v - v.rolling(24).mean()) / (v.rolling(24).std() + 1e-9)
    f["volume_delta_6"] = np.log((v + 1) / (v.shift(6) + 1))
    vwap20 = (c * v).rolling(20).sum() / (v.rolling(20).sum() + 1e-9)
    f["vwap_dev_20"] = (c - vwap20) / (vwap20 + 1e-9)
    # --- MICROSTRUCTURE (candle shape -- straight from user's module) ---
    rng = (h - l).clip(lower=1e-9)
    f["spread_proxy"] = rng / c.abs().clip(lower=1e-9)
    f["body_pct"] = (c - o).abs() / rng
    upper = (h - np.maximum(o, c)).clip(lower=0)
    lower = (np.minimum(o, c) - l).clip(lower=0)
    f["upper_wick_pct"] = upper / rng
    f["lower_wick_pct"] = lower / rng
    f["wick_imbalance"] = (upper - lower) / rng
    sv = np.sign(lr).fillna(0) * v
    f["volume_imbalance"] = sv / (sv.abs().rolling(20).mean() + 1e-9)
    sd = lr.rolling(10).std()
    f["vol_compression"] = sd / (sd.rolling(50).mean() + 1e-9)
    f["realized_vol_12"] = np.sqrt((lr ** 2).rolling(12).sum())
    # cyclic time-of-day
    hod = d.index.hour.values
    f["tod_sin"] = np.sin(2 * np.pi * hod / 24)
    f["tod_cos"] = np.cos(2 * np.pi * hod / 24)
    # regime-tagging vol (kept name 'vol')
    f["vol"] = lr.rolling(24).std()
    return f


# Targets

def target_vol_regime(close: pd.Series, horizon=12, lookback=24 * 30):
    lr = np.log(close / close.shift(1))
    fut_vol = lr.rolling(horizon).std().shift(-horizon)        # next-12h realized vol
    thr = fut_vol.rolling(lookback, min_periods=24 * 5).quantile(0.667)
    y = (fut_vol > thr).astype(float)
    y[thr.isna() | fut_vol.isna()] = np.nan
    return y


def target_large_move(close: pd.Series, horizon=12, lookback=24 * 30, q=0.70):
    absret = np.abs(np.log(close.shift(-horizon) / close))
    thr = absret.rolling(lookback, min_periods=24 * 5).quantile(q)
    y = (absret > thr).astype(float)
    y[thr.isna() | absret.isna()] = np.nan
    return y


def target_tb_dir(d: pd.DataFrame, tp_atr=2.0, sl_atr=1.5, horizon=12, cost_bps=5.0):
    """User's triple-barrier label (long side), ATR-scaled, after-cost.
    +1 if TP hit before SL, 0 otherwise. NaN where horizon overflows."""
    c = d.close.values; h = d.high.values; l = d.low.values
    atr = (_atr(d.high, d.low, d.close, 14)).values
    cost = cost_bps / 1e4
    n = len(c); y = np.full(n, np.nan)
    for t in range(n - horizon):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        tp = c[t] + tp_atr * atr[t]
        sl = c[t] - sl_atr * atr[t]
        tp_eff = tp * (1 + cost)
        lab = 0
        for k in range(1, horizon + 1):
            if h[t + k] >= tp_eff: lab = 1; break
            if l[t + k] <= sl: lab = 0; break
        y[t] = lab
    return pd.Series(y, index=d.index)


if __name__ == "__main__":
    d = load_klines("data/raw/BTCUSDT_1h.csv")
    f = build_features(d).replace([np.inf, -np.inf], np.nan)
    print("features:", f.shape, "| valid rows after dropna:", f.dropna().shape[0])
    yv = target_vol_regime(d.close); ym = target_large_move(d.close); yt = target_tb_dir(d)
    for nm, y in [("vol_regime", yv), ("large_move", ym), ("tb_dir", yt)]:
        yy = y.dropna()
        print(f"  {nm:11s}: n={len(yy)}  pos_rate={yy.mean():.3f}")
