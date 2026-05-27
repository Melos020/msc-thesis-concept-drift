# Raw data

The hourly OHLCV datasets are not redistributed in this repository (size, and
to respect the data source's terms). Place them here before a Level-2 full
rerun. Level-1 reproduction (recomputing the corrected tables) does not need
them — it uses the per-window probability dumps in `results/provenance/`.

## Expected files

```
data/raw/BTCUSDT_1h.csv
data/raw/ETHUSDT_1h.csv
data/raw/BNBUSDT_1h.csv
```

The loader (`data/features_hourly.py::load_klines`) also accepts `BTC_1h.csv`
or any `BTC*.csv` in this directory; set `DATA_DIR` to point elsewhere.

## Expected schema

CSV with a header, one row per hourly bar, chronological:

```
timestamp,open,high,low,close,volume
2018-01-01 00:00:00,13715.0,13720.0,13680.0,13700.0,1234.5
...
```

- `timestamp` parseable by pandas (ISO-8601 or epoch ms).
- Spot klines, USDT pairs, hourly interval, spanning roughly 2018–2025.
- Approximately 30,000 rows per asset after the downsample-by-2 in the
  pipeline (about 60,000 raw hourly bars).

## Download recipe (Binance public klines)

The series were built from Binance public spot klines. One reproducible route:

1. Download monthly 1h kline archives for `BTCUSDT`, `ETHUSDT`, `BNBUSDT` from
   the Binance public data portal (`data.binance.vision`), interval `1h`.
2. Concatenate per asset in chronological order.
3. Keep the first six columns and rename to
   `timestamp, open, high, low, close, volume`.
4. Save as `data/raw/{ASSET}USDT_1h.csv`.

## Integrity

After assembling each file, record a checksum so an examiner can confirm an
identical input:

```bash
sha256sum data/raw/*.csv > data/raw/CHECKSUMS.sha256
```

Commit `CHECKSUMS.sha256` (not the CSVs). The pipeline is deterministic given
identical inputs at Level 1; at Level 2, retraining introduces small
seed/hardware variation as noted in docs/REPRODUCE.md.
