"""Real OHLCV loader for BTC-USD and ETH-USD daily data, 2018-2024.

Source: Yahoo Finance OHLCV (user-supplied CSVs uploaded 2026-05-11).
Both assets verified to have:
  - Date range 2018-01-01 -> 2024-12-31 (2,557 days)
  - No missing dates, no NaN values
  - OHLC invariants: High >= Open, Close, Low for every row
  - Non-zero Volume for every row

Replaces the prior FRED close-only loader. Restores the original thesis
feature set with TRUE ATR computed from High/Low/Close.
"""
import pandas as pd
from typing import Literal


def load_ohlcv(asset: Literal['BTC', 'ETH'],
               start: str = '2018-01-01',
               end: str = '2024-12-31',
               data_dir: str = 'data') -> pd.DataFrame:
    """Load cleaned OHLCV daily series for BTC or ETH.

    Returns DataFrame indexed by date with columns: open, high, low, close, volume.
    """
    path = f'{data_dir}/{asset}-USD_OHLCV_clean.csv'
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    df = df.loc[start:end].copy()
    for c in ['open', 'high', 'low', 'close', 'volume']:
        df[c] = df[c].astype(float)
    return df


# Backward-compat shim for the FRED close-only loader (kept for reproducibility
# of the close-only validation report).
def load_btc_real(path: str = 'data/BTC-USD_close_clean.csv',
                  start: str = '2018-01-01',
                  end: str = '2024-12-31') -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date']).set_index('date')
    df = df.loc[start:end].copy()
    df['close'] = df['close'].astype(float).ffill().bfill()
    return df


if __name__ == '__main__':
    for asset in ['BTC', 'ETH']:
        df = load_ohlcv(asset)
        print(f'\n=== {asset}-USD OHLCV ===')
        print(f'  rows:        {len(df)}')
        print(f'  range:       {df.index.min().date()} to {df.index.max().date()}')
        print(f'  close min:   ${df["close"].min():,.2f}')
        print(f'  close max:   ${df["close"].max():,.2f}')
        print(f'  avg volume:  {df["volume"].mean():,.0f}')
