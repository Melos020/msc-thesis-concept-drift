"""Feature engineering (thesis Ch 3.3).

Standard features:
  - RSI(14)
  - MACD(12, 26, 9) -> normalized histogram
  - Bollinger Band width(20, 2) (normalized by middle band)
  - log returns
  - ATR(14) if OHLC available (TRUE ATR), else vol_proxy fallback

The vol_proxy fallback is preserved for the deprecated close-only validation
path. With OHLCV available (BTC and ETH OHLCV CSVs, 2018-2024), true ATR is
used and the substitution caveat no longer applies.
"""
import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period, min_periods=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_width(close: pd.Series, period: int = 20, n_std: float = 2.0) -> pd.Series:
    ma = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = ma + n_std * std
    lower = ma - n_std * std
    return (upper - lower) / ma


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """True Average True Range. Requires OHLC."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def vol_proxy(close: pd.Series, period: int = 14) -> pd.Series:
    """Close-only volatility proxy (deprecated; kept for FRED close-only path)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(period, min_periods=period).std() * np.sqrt(period) * close


def build_features(df: pd.DataFrame, data_source: str = 'synthetic') -> pd.DataFrame:
    """Build the 5-feature input tensor used by both models.

    Features (after dropna):
      - rsi:       RSI(14) normalized to [0, 1]
      - macd_hist: MACD(12, 26, 9) histogram, scaled by close
      - bb_width:  Bollinger band width, normalized by middle band
      - log_return: 1-day log return
      - vol:       ATR(14) / close (true ATR if OHLC available; else vol_proxy)

    data_source in {'synthetic', 'btc_ohlcv', 'eth_ohlcv', 'btc_real_close'}.
    """
    out = pd.DataFrame(index=df.index)
    out['rsi'] = rsi(df['close'], 14) / 100.0
    _, _, hist = macd(df['close'], 12, 26, 9)
    out['macd_hist'] = hist / df['close']
    out['bb_width'] = bollinger_width(df['close'], 20, 2.0)
    out['log_return'] = np.log(df['close'] / df['close'].shift(1))
    has_ohlc = {'high', 'low'}.issubset(df.columns)
    if has_ohlc:
        out['vol'] = atr(df['high'], df['low'], df['close'], 14) / df['close']
    else:
        out['vol'] = vol_proxy(df['close'], 14) / df['close']
    return out


def featurize(df: pd.DataFrame, data_source: str = 'synthetic') -> pd.DataFrame:
    """Convenience: build features and drop initial NaNs from warm-up windows."""
    return build_features(df, data_source=data_source).dropna()


if __name__ == '__main__':
    from .btc_real import load_ohlcv
    for asset in ['BTC', 'ETH']:
        df = load_ohlcv(asset)
        f = featurize(df, data_source=f'{asset.lower()}_ohlcv')
        print(f'\n{asset} features shape: {f.shape}')
        print(f.describe().round(4))
