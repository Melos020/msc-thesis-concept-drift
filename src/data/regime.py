"""Causal regime segmentation (thesis Ch 3.3).

Uses an expanding-window volatility-quantile procedure with rolling-mode
smoothing. The quantile at time t is computed over the trailing window
[0, t], so no future information leaks into the regime label.
"""
import numpy as np
import pandas as pd


def _rolling_mode(s: pd.Series, window: int) -> pd.Series:
    """Causal rolling mode (most-frequent regime in trailing window)."""
    vals = s.values
    out = vals.copy()
    for t in range(len(vals)):
        lo = max(0, t - window + 1)
        win = vals[lo:t + 1]
        # mode
        u, c = np.unique(win, return_counts=True)
        out[t] = u[c.argmax()]
    return pd.Series(out, index=s.index)


def causal_vol_quantile_regime(vol: pd.Series,
                               n_regimes: int = 4,
                               min_history: int = 60,
                               smooth_window: int = 60) -> pd.Series:
    """Assign each timestep to a regime based on the EXPANDING quantile of vol.
    
    For timestep t, computes quantile-bins from vol[:t+1] (causal, no leak).
    Optional smoothing by rolling mode to suppress noisy single-day flips.
    """
    regimes = np.full(len(vol), -1, dtype=np.int64)
    v = vol.values
    for t in range(len(v)):
        if t < min_history or np.isnan(v[t]):
            regimes[t] = 0
            continue
        history = v[:t + 1]
        history = history[~np.isnan(history)]
        if len(history) < min_history:
            regimes[t] = 0
            continue
        bins = np.quantile(history, np.linspace(0, 1, n_regimes + 1))
        b = np.searchsorted(bins[1:-1], v[t])
        regimes[t] = b
    out = pd.Series(regimes, index=vol.index, name='regime')
    if smooth_window > 1:
        out = _rolling_mode(out, smooth_window)
    return out


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    vol = pd.Series(np.abs(rng.normal(0, 1, 500)))
    r = causal_vol_quantile_regime(vol, n_regimes=4)
    print(r.value_counts())
