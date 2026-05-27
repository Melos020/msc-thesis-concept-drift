# Reproduction guide

Two levels of reproduction are supported: recompute the corrected tables from
the shipped provenance (fast, no GPU, no retraining), or rerun the full
walk-forward training from raw data (slow, needs torch).

## Environment

```bash
python --version          # 3.10 or 3.11 recommended
pip install -r requirements.txt
# or: conda env create -f environment.yml && conda activate tcn-anfis-drift
```

The corrected run was performed with Python 3.x and PyTorch on CPU/GPU; the
correction layer itself is pure NumPy/SciPy/scikit-learn and needs no GPU.

## Level 1 — recompute corrected tables from provenance (minutes, no torch)

Every corrected table is a deterministic function of the per-window
probability dumps in `results/provenance/`. To regenerate them:

```bash
# point the corrections script at the provenance dumps
mkdir -p results_hourly
cp results/provenance/*_window_probs.pkl results_hourly/
python run_all_corrections.py hourly results_hourly
# -> corrections_out/threshold_fair_summary.csv, threshold_fair_paired.csv,
#    inference_corrected.csv, DECISION.txt
```

Compare `corrections_out/threshold_fair_summary.csv` against
`results/final/threshold_fair_summary.csv`; they must match. For the per-asset
and drift breakdown:

```bash
python run_full_corrections.py results_hourly
# -> corrections_out/per_asset_threshold_fair.csv, etc.
```

## Level 2 — full rerun from raw hourly data (hours, needs torch)

1. Obtain hourly OHLCV CSVs for BTC, ETH, BNB (see `data/raw/README.md` for
   format, filenames, and a download recipe).

2. Run the walk-forward pipeline (trains TCN + ANFIS, runs HAR + GARCH, dumps
   per-window probabilities):

```bash
DATA_DIR=data/raw python run_hourly_patched.py BTC vol_regime
DATA_DIR=data/raw python run_hourly_patched.py ETH vol_regime
DATA_DIR=data/raw python run_hourly_patched.py BNB vol_regime
```

Each asset writes `results_hourly/{ASSET}_vol_regime_*` including a fresh
`*_window_probs.pkl`.

3. Apply the corrections (same as Level 1 step):

```bash
python run_all_corrections.py hourly results_hourly
```

## Expected output

```
model   n    AUC  MCC_at_0p5  MCC_oracle  MCC_heldout_youden  MCC_heldout_base
  tcn 213 0.6810      0.2473      0.3142              0.2356            0.2506
anfis 213 0.6413      0.0139      0.2659              0.1894            0.2020
  har 213 0.5551      0.0000      0.0037              0.0631            0.0674
garch 213 0.6155      0.0140      0.2201              0.1364            0.1512
```

Small numerical differences in a full retrain (Level 2) are expected from
nondeterministic GPU kernels and seed effects; the ordering and the modest
effect size are stable. Level 1 is exactly reproducible because it does not
retrain.

## Determinism notes

- All preprocessing (scaler, mutual-information feature selection, ANFIS
  premise k-means) is fit on training data only, inside each window.
- TCN seeds are averaged at the window level; two seeds by default.
- The threshold-fair held-out thresholds are chosen only on the pre-drift
  segment, never on the evaluation segment.
