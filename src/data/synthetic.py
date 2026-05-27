"""Synthetic non-stationary time-series generator (thesis Ch 3.2)."""
import numpy as np
import pandas as pd


def generate_synthetic_series(n_days: int = 2557, seed: int = 0) -> pd.DataFrame:
    """Generate a synthetic OHLC-like price series with regime structure.
    
    Returns DataFrame with columns: date, open, high, low, close, volume.
    """
    rng = np.random.default_rng(seed)
    # Geometric Brownian motion baseline with regime-dependent drift and vol
    log_returns = np.zeros(n_days)
    # 4 regimes cycling: low-vol up, low-vol down, high-vol up, high-vol down
    block = n_days // 4
    for k in range(4):
        start = k * block
        end = (k + 1) * block if k < 3 else n_days
        mu = [0.0008, -0.0006, 0.0012, -0.0010][k]
        sigma = [0.018, 0.020, 0.045, 0.050][k]
        log_returns[start:end] = rng.normal(mu, sigma, end - start)
    
    close = 10000 * np.exp(np.cumsum(log_returns))
    # Synthetic OHL via intraday range
    intraday_range = np.abs(rng.normal(0, 0.008, n_days)) * close
    high = close + intraday_range
    low = close - intraday_range * 0.7
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.exp(rng.normal(15, 1, n_days))
    
    dates = pd.date_range(start='2018-01-01', periods=n_days, freq='D')
    return pd.DataFrame({
        'date': dates,
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume,
    }).set_index('date')


if __name__ == '__main__':
    df = generate_synthetic_series(seed=0)
    print(df.describe())
