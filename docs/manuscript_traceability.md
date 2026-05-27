# Manuscript traceability

Every major claim, table, and figure in the final manuscript, mapped to the
script that produces it and the output file it lives in. Paths are relative to
the repository root.

## Headline result (Abstract, Results §"The learnable target")

> TCN held-out MCC 0.236 vs ANFIS 0.189; oracle 0.314 vs 0.266; AUC 0.681 vs
> 0.641; HAR 0.063, GARCH 0.136; d ≈ 0.35–0.45; 213 windows.

- Produced by: `run_all_corrections.py hourly results_hourly`
- From provenance: `results/provenance/{BTC,ETH,BNB}_vol_regime_window_probs.pkl`
- Output: `results/final/threshold_fair_summary.csv`,
  `results/final/threshold_fair_paired.csv`, `results/final/DECISION.txt`
- Evaluation logic: `robustness_core.py`
  (`threshold_fair_row`, `best_threshold_oracle`, `threshold_from_heldout`)

## Table 5 — Threshold-fair results (pooled, 213 windows)

- Script: `run_all_corrections.py` -> `produce_corrections`
- Output: `results/final/threshold_fair_summary.csv`
- Per-asset breakdown: `results/final/per_asset_threshold_fair.csv`
  (`run_full_corrections.py`)

## Table 6 — Paired TCN-vs-ANFIS contrast

- Script: `run_all_corrections.py` (`paired_wilcoxon` in
  `robustness_core.py`)
- Output: `results/final/threshold_fair_paired.csv`

## Table 6b — Overlap-aware significance (naive / non-overlap / Nadeau–Bengio)

- Script: `run_all_corrections.py` -> `full_inference`
- Logic: `robustness_core.py` (`nadeau_bengio_t`,
  `nonoverlapping_indices`)
- Output: `results/final/inference_corrected.csv`

## Per-asset summaries / paired tests / drift (BTC, ETH, BNB)

- Script: `run_hourly_patched.py {ASSET} vol_regime`
- Output: `results/tables/{ASSET}_vol_regime_summary.csv`,
  `..._paired.csv`, `..._decomp.csv`

## Daily-direction null (Results §"The corrected null on daily direction")

- Logic: `corrected_pipeline.py` (held-out walk-forward) with the
  near-unpredictable control series in `src/data/synthetic.py`.
- Note: the daily-direction task is the negative control, not a headline
  result; it establishes that the pipeline does not manufacture signal. The
  daily input series themselves are not redistributed in this package.

## Figures — file names vs final-manuscript numbers

The architecture/design figures kept their original draft file names; the final
manuscript renumbered them. The mapping is:

- `results/figures/fig7_tcn_architecture.png` — TCN architecture (manuscript Fig 2)
- `results/figures/fig8_anfis_architecture.png` — ANFIS architecture (manuscript Fig 3)
- `results/figures/fig9_walkforward.png` — walk-forward design (manuscript Fig 4)
- `results/figures/fig10_threshold_fair.png` — threshold-fair / overlap-aware
  evaluation schematic (manuscript Fig 8)

The SVGs (`tcn_fig.svg`, `anfis_fig.svg`, `wf_fig.svg`, `tf_fig.svg`) are the
editable sources for these four renders. The remaining manuscript figures
(the forest plots and paired-distribution plots, Figs 5–7 and 9) are embedded
directly in the thesis document under `thesis/`.

## Figure: TCN architecture (manuscript Fig 2)

- Source: `results/figures/tcn_fig.svg` -> `fig7_tcn_architecture.png`
- Reflects: `src/models/tcn.py` (3 residual blocks, dilations 1/2/4, kernel 3,
  64 channels, dropout 0.15) and `run_enriched.py` (Adam, lr 1e-3,
  early stopping, seq_len 24, 2 seeds averaged).

## Figure: ANFIS architecture (manuscript Fig 3)

- Source: `results/figures/anfis_fig.svg` -> `fig8_anfis_architecture.png`
- Reflects: `src/models/anfis.py` (Gaussian MFs, k-means premise init,
  ridge-LSE consequents λ=1e-3, 6 MI-selected features × 3 MFs).

## Figure: walk-forward design (manuscript Fig 4)

- Source: `results/figures/wf_fig.svg` -> `fig9_walkforward.png`
- Reflects: `corrected_pipeline.py` (`make_rolling_windows`: window 2400,
  step 400, train 0.60, held-out pre 0.13).

## Figure: threshold-fair / overlap-aware evaluation (manuscript Fig 8)

- Source: `results/figures/tf_fig.svg` -> `fig10_threshold_fair.png`
- Reflects: `robustness_core.py` (the three threshold readings
  and the three inference views).

## HAR / GARCH baseline result (Discussion, Threats §6)

> HAR held-out MCC 0.063 / AUC 0.555; GARCH 0.136 / 0.616; both clearly beaten
> by TCN and ANFIS.

- Script: `baselines_vol.py` (`har_vol_regime`,
  `garch_vol_regime`), invoked inside `run_hourly_patched.py`
- Output: in the per-asset summaries and the pooled
  `results/final/threshold_fair_summary.csv`

## Degradation-ratio decomposition (Methodology, the four corrections)

- Logic: `corrected_pipeline.py` (`corrected_DR`)
- The held-out pre-drift baseline vs the post segment; recorded per window in
  the `*_windows.csv` files under `results/provenance/`.

## What is intentionally NOT in this repository

- The earlier five-task breadth table and the overlap-inflated drift table:
  removed from the final manuscript, excluded here to avoid contradictory
  outputs. Only the corrected primary-task (vol_regime) results are retained.
- Superseded scripts (`run_ckpt.py`, `aggregate_hourly.py`, unpatched
  `run_hourly.py`, early `threshold_robustness.py`, the synthetic smoke-test
  `validate_baselines.py`): excluded as obsolete.
