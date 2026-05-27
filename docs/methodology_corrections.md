# Methodology corrections

This project began with an evaluation that produced inflated findings. The
corrected pipeline in this repository addresses four classes of pitfall. This
document explains each, why it matters, and where it is handled in code.

## 1. In-sample pre-drift baseline (degradation ratio inflation)

**Pitfall.** Measuring "before drift" performance on data the model was trained
on makes the drop to out-of-sample look like drift, when most of it is ordinary
overfitting. On the daily tree ensembles this overstated degradation roughly
tenfold.

**Fix.** A held-out pre-drift segment that is never used for fitting. The
degradation ratio is an out-of-sample-vs-out-of-sample contrast.

**Code.** `corrected_pipeline.py` — `make_rolling_windows` (train / pre-holdout
/ post split), `corrected_DR`.

## 2. Seed-level and window-level pseudo-replication

**Pitfall.** Treating many seeds on one split, or many overlapping windows, as
independent observations fabricates statistical power: the standard error is
computed as if the effective sample were far larger than it is.

**Fix.** Seeds are averaged within a window before any test. Inference is then
performed across windows, and additionally:
- restricted to a maximal non-overlapping subsample (genuinely independent), and
- corrected with the Nadeau–Bengio resampled t-test, which inflates the
  variance to account for train/test overlap.

The reported significance is the conservative reading across these, not the
smallest p-value.

**Code.** `robustness_core.py` — `nadeau_bengio_t`,
`nonoverlapping_indices`, `full_inference`; `window_stats.py`.

## 3. Fixed-threshold metric conflates ranking with calibration

**Pitfall.** MCC at a fixed 0.5 cut mixes two different things: how well a model
ranks cases, and whether its probabilities happen to straddle that cut. A
well-ranking but poorly-calibrated model (ANFIS here) can score near zero MCC at
0.5 while ranking almost as well as the TCN.

**Fix.** A threshold-fair reading of every model:
- **AUC** — threshold-free ranking, the fair head-to-head.
- **MCC oracle** — best MCC over a threshold grid on the evaluation window; a
  capability ceiling, not deployable, reported so neither model is penalised by
  an arbitrary cut.
- **MCC held-out** — threshold chosen only on the held-out pre-drift window
  (base rate and Youden's J); the honest deployable number.

**Code.** `robustness_core.py` — `threshold_fair_row`,
`best_threshold_oracle`, `threshold_from_heldout`.

## 4. Unlearnable target

**Pitfall.** Forecasting next-day price direction targets a quantity that is
close to a martingale difference at short horizons; with no signal, no
architecture can distinguish itself and any reported difference is noise or
leakage.

**Fix.** Re-target onto the next-12h volatility regime, which is autocorrelated
and genuinely forecastable, so that an architectural comparison is meaningful.
The daily-direction task is retained only as a negative control.

**Code.** `features_hourly.py` (`target_vol_regime`); the daily null is the
control established with `corrected_pipeline.py` and `src/data/synthetic.py`.

## The corrected scientific position

Putting these together changes the interpretation rather than overturning it.
The TCN keeps a **modest but consistent** advantage over ANFIS on the
volatility-regime task: comparable ranking, better calibration, and better
decisions at a fixed operating threshold, preserved under regime shift. Both
learned models clearly beat the HAR and GARCH baselines. The naive fixed-0.5
effect size (d ≈ 1.6) shrinks to d ≈ 0.35–0.45 once the threshold and the window
overlap are handled correctly, and only the fixed-0.5 gap stays significant
under the strictest overlap correction. That reduction is a gain in
credibility: it is what the evidence supports once the measurement is done
honestly.
