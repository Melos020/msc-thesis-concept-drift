# A note on file organization

The pipeline uses a flat module layout. The entry-point scripts at the
repository root import each other as top-level modules and import the `src/`
package:

```
run_hourly_patched.py     walk-forward run (TCN+ANFIS+HAR+GARCH per asset)
run_all_corrections.py    corrections, pooled (threshold-fair + overlap-aware)
run_full_corrections.py   per-asset / per-task / drift corrected tables
run_enriched.py           TCN trainer (train_tcn_strong / predict_tcn_strong)
corrected_pipeline.py     windows, metrics, calibration, degradation ratio
robustness_core.py        threshold-fair eval + Nadeau-Bengio + non-overlap
baselines_vol.py          HAR (Corsi) and GARCH(1,1) baselines
features_hourly.py        causal OHLCV feature engine + targets
window_stats.py           window-level aggregation helpers
src/                      importable package (TCN, ANFIS, regime, training)
```

Run everything from the repository root so the flat imports plus the `src/`
package resolve without any path juggling.
