# TCN vs ANFIS under Concept Drift — Reproducibility Package

Reproducibility code and results for the MSc thesis *Comparing
Temporal Convolutional Networks and ANFIS under Concept Drift in Cryptocurrency
Volatility Forecasting: A Leakage-Free Walk-Forward Evaluation Framework*.

This repository contains the final state of the project.

## Start

```bash
pip install -r requirements.txt
python run_all_corrections.py hourly results/provenance
```

Main corrected outputs are written to:
- `results/final/`
- `results/provenance/`
- `figures/`

Notes:
- Experiments were originally executed on consumer NVIDIA GPU hardware, but the correction and evaluation scripts also run on CPU (slower).
- Small metric variations across hardware/platforms are normal despite fixed seeds.
- Raw Binance OHLCV data is not redistributed in this repository and must be downloaded separately following the provided instructions.

## Claims of the Thesis

The corrected scientific position is deliberately modest and is the position
this repository supports:

- **Daily directional prediction collapses toward a null** under leakage-free
  evaluation. Neither architecture, nor five strong baselines, shows an
  out-of-sample edge on next-day price direction.
- **Volatility-related targets remain partially learnable.** The next-12h
  volatility regime is autocorrelated and forecastable, which is what makes an
  architectural comparison meaningful.
- **Overlap dependence inflates naive significance.** Rolling windows that
  overlap are not independent; naive tests over them overstate significance.
- **Threshold choice changes interpretation.** A fixed 0.5 cut conflates
  ranking ability with calibration. Evaluated fairly, much of the apparent gap
  is a calibration effect.
- **The TCN holds a modest but consistent advantage over ANFIS** on the
  volatility-regime task under corrected, threshold-fair, overlap-aware
  evaluation, and **both learned models clearly beat the classical HAR and
  GARCH baselines.**

The headline corrected numbers (213 leakage-free walk-forward windows pooled
over BTC, ETH, BNB):

| model | AUC | MCC @0.5 | MCC oracle | MCC held-out (Youden) |
|-------|-----|----------|------------|------------------------|
| TCN   | 0.681 | 0.247 | 0.314 | 0.236 |
| ANFIS | 0.641 | 0.014 | 0.266 | 0.189 |
| HAR   | 0.555 | 0.000 | 0.004 | 0.063 |
| GARCH | 0.616 | 0.014 | 0.220 | 0.136 |

Effect size TCN vs ANFIS under an honest held-out threshold: Cohen's d ≈ 0.35.
Under the most conservative overlap correction (Nadeau–Bengio) only the
fixed-0.5 MCC gap remains independently significant; the ranking and
held-out-threshold gaps stay directionally in the TCN's favour but are not
significant once window dependence is fully accounted for. The corrected protocol substantially reduced the originally inflated effect sizes, yielding more conservative and more reliable estimates.

## Repository layout

```
README.md                     this file
requirements.txt              pip dependencies (pinned)
environment.yml               conda environment
LICENSE                       MIT
.gitignore

thesis/
  Mathieu_Ajaka_MSc_Thesis.pdf   the final manuscript
  Mathieu_Ajaka_MSc_Thesis.docx  editable source

# --- runnable pipeline (flat layout; run from the repo root) ---
run_hourly_patched.py         full walk-forward run: TCN+ANFIS+HAR+GARCH per asset
run_all_corrections.py        threshold-fair + overlap-aware corrections (pooled)
run_full_corrections.py       per-asset / per-task / drift corrected tables
run_enriched.py               TCN trainer (train_tcn_strong / predict_tcn_strong)
corrected_pipeline.py         windows, metrics, calibration, degradation ratio
robustness_core.py            threshold-fair eval + Nadeau-Bengio + non-overlap
baselines_vol.py              HAR (Corsi) and GARCH(1,1) volatility baselines
features_hourly.py            causal OHLCV feature engine + targets
window_stats.py               window-level aggregation helpers

src/                          importable package the scripts depend on
  models/tcn.py               Temporal Convolutional Network
  models/anfis.py             Adaptive Neuro-Fuzzy Inference System
  data/regime.py              causal volatility-quantile regime segmentation
  data/features.py            feature builder (daily-direction control path)
  data/synthetic.py           near-unpredictable synthetic series (audit control)
  data/btc_real.py            OHLCV close loader (daily-direction control path)
  data/drift_sequences.py     drift sequence helpers
  evaluation/metrics.py       MCC, AUC, ECE, balanced accuracy
  evaluation/statistics.py    bootstrap / paired-test helpers
  training/train.py           TCN train/predict/windowing used by the pipeline

data/
  raw/README.md               where to place hourly CSVs + schema + download recipe

results/
  final/                      corrected pooled tables + DECISION.txt
  tables/                     per-asset summaries, paired tests, drift decomp
  figures/                    manuscript architecture/design figures (PNG + SVG source)
  provenance/                 per-window probability dumps (*.pkl): the raw
                              material every corrected table is computed from

docs/
  manuscript_traceability.md  every table/figure -> script -> output mapping
  REPRODUCE.md                exact reproduction commands
  methodology_corrections.md  the evaluation pitfalls addressed and how
```

The pipeline uses a flat module layout: the entry-point scripts import each
other as top-level modules and import the `src/` package. Run everything from
the repository root so these imports resolve without any path juggling.

## Quick start

```bash
pip install -r requirements.txt

# Level 1 — recompute the corrected tables from the shipped provenance
# (minutes, no torch, no retraining):
mkdir -p results_hourly
cp results/provenance/*_window_probs.pkl results_hourly/
python run_all_corrections.py hourly results_hourly

# Level 2 — full rerun from raw hourly data (hours, needs torch):
# place hourly CSVs in data/raw/ (see data/raw/README.md), then:
python run_hourly_patched.py BTC vol_regime
python run_hourly_patched.py ETH vol_regime
python run_hourly_patched.py BNB vol_regime
python run_all_corrections.py hourly results_hourly
```

Full instructions: `docs/REPRODUCE.md`. Claim-to-code mapping:
`docs/manuscript_traceability.md`.

## Provenance note

The corrected tables in `results/final/` are computed directly from the
per-window probability dumps in `results/provenance/` by
`run_all_corrections.py`. Those dumps are the genuine output of the
walk-forward run on hourly Binance data for BTC, ETH and BNB. An examiner can
recompute every corrected table from the provenance files without retraining
any model, or retrain from scratch with the commands above.
