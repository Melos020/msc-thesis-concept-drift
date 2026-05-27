"""
run_full_corrections.py  --  complete, consistent corrected tables.

The basic run_all_corrections.py pools all windows for the primary task. For a
manuscript, every table must use the SAME corrected inference (threshold-fair +
overlap-aware), so this script regenerates:

  * PER-ASSET threshold-fair summary (BTC / ETH / BNB separately)
  * POOLED threshold-fair summary
  * PER-TASK summary for every task whose probability dump exists
  * DRIFT DECOMPOSITION (same-regime vs regime-shift) under corrected inference
  * Overlap-corrected inference (naive / non-overlap / Nadeau-Bengio) per metric

It reads the *_window_probs.pkl dumps written by run_hourly_patched.py. To cover
more tasks, first generate their dumps, e.g.:
    python run_hourly_patched.py BTC vol_regime
    python run_hourly_patched.py BTC large_move
    python run_hourly_patched.py BTC tb_dir
(and ETH, BNB). Then:
    python run_full_corrections.py results_hourly

The dump filename encodes asset and task: {ASSET}_{task}_window_probs.pkl, and
each row carries a regime_changed flag if present (added by the patched driver),
so the drift split needs no extra data.

Outputs (corrections_out/):
    per_asset_threshold_fair.csv
    per_task_threshold_fair.csv
    drift_decomposition_corrected.csv
    inference_corrected_full.csv
    DECISION_FULL.txt
"""
from __future__ import annotations
import os, sys, glob, re
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_core import threshold_fair_row, full_inference, paired_wilcoxon

OUT = "corrections_out"
os.makedirs(OUT, exist_ok=True)


def load_dumps(results_dir):
    """Return a dict {(asset,task): DataFrame} from every *_window_probs.pkl."""
    out = {}
    for pk in sorted(glob.glob(os.path.join(results_dir, "*_window_probs.pkl"))):
        base = os.path.basename(pk).replace("_window_probs.pkl", "")
        # base looks like  BTC_vol_regime  ;  asset is first token
        m = re.match(r"([A-Za-z0-9]+)_(.+)", base)
        asset, task = (m.group(1), m.group(2)) if m else (base, "vol_regime")
        out[(asset, task)] = pd.read_pickle(pk)
    return out


def tf_table(df):
    """threshold-fair per-window rows -> per-model means."""
    rows = []
    for _, r in df.iterrows():
        row = threshold_fair_row(r.p_pre, r.y_pre, r.p_post, r.y_post, r.train_pos_rate)
        row.update({"wid": r.wid, "model": r.model,
                    "regime_changed": int(r.get("regime_changed", -1))})
        rows.append(row)
    tf = pd.DataFrame(rows)
    summ = (tf.groupby("model")
              .agg(n=("wid", "count"),
                   AUC=("auc", "mean"),
                   MCC_0p5=("mcc_at_0.5", "mean"),
                   MCC_oracle=("mcc_oracle", "mean"),
                   MCC_heldout_youden=("mcc_heldout_youden", "mean"),
                   MCC_heldout_base=("mcc_heldout_base", "mean"))
              .round(4).reset_index())
    return tf, summ


def main(results_dir="results_hourly"):
    dumps = load_dumps(results_dir)
    if not dumps:
        print(f"No dumps in {results_dir}. Run run_hourly_patched.py first.")
        sys.exit(1)

    order = ["tcn", "anfis", "har", "garch"]

    # PER-ASSET (primary task = vol_regime if present)
    per_asset = []
    primary = "vol_regime"
    for (asset, task), df in dumps.items():
        if task != primary:
            continue
        _, s = tf_table(df)
        s.insert(0, "asset", asset)
        per_asset.append(s)
    if per_asset:
        pa = pd.concat(per_asset, ignore_index=True)
        # pooled
        allp = pd.concat([df for (a, t), df in dumps.items() if t == primary], ignore_index=True)
        _, sp = tf_table(allp); sp.insert(0, "asset", "POOLED")
        pa = pd.concat([pa, sp], ignore_index=True)
        pa["model"] = pd.Categorical(pa["model"], order)
        pa = pa.sort_values(["asset", "model"]).reset_index(drop=True)
        pa.to_csv(f"{OUT}/per_asset_threshold_fair.csv", index=False)
        print("=== PER-ASSET threshold-fair (primary task) ===")
        print(pa.to_string(index=False)); print()

    # PER-TASK (pooled across assets) 
    per_task = []
    tasks = sorted({t for (_, t) in dumps})
    for task in tasks:
        allp = pd.concat([df for (a, t), df in dumps.items() if t == task], ignore_index=True)
        _, s = tf_table(allp); s.insert(0, "task", task)
        per_task.append(s)
    pt = pd.concat(per_task, ignore_index=True)
    pt["model"] = pd.Categorical(pt["model"], order)
    pt = pt.sort_values(["task", "model"]).reset_index(drop=True)
    pt.to_csv(f"{OUT}/per_task_threshold_fair.csv", index=False)
    print("=== PER-TASK threshold-fair (pooled assets) ===")
    print(pt.to_string(index=False)); print()

    # DRIFT DECOMPOSITION (primary task, corrected) 
    allp = pd.concat([df for (a, t), df in dumps.items() if t == primary], ignore_index=True)
    tf, _ = tf_table(allp)
    if (tf.regime_changed >= 0).any():
        drift = []
        for model in order:
            sub = tf[tf.model == model]
            for lbl, msk in [("same_regime", sub.regime_changed == 0),
                             ("regime_shift", sub.regime_changed == 1)]:
                g = sub[msk]
                if len(g):
                    drift.append({"model": model, "transition": lbl, "n": len(g),
                                  "AUC": round(g.auc.mean(), 4),
                                  "MCC_0p5": round(g["mcc_at_0.5"].mean(), 4),
                                  "MCC_heldout_youden": round(g.mcc_heldout_youden.mean(), 4)})
        dd = pd.DataFrame(drift)
        dd.to_csv(f"{OUT}/drift_decomposition_corrected.csv", index=False)
        print("=== DRIFT DECOMPOSITION (corrected, primary task) ===")
        print(dd.to_string(index=False)); print()
    else:
        print("(no regime_changed flag in dumps -> re-run patched driver to enable drift split)\n")

    # OVERLAP-CORRECTED INFERENCE (primary, TCN vs ANFIS) 
    tf, _ = tf_table(allp)
    wl = int(allp.window_len.iloc[0]); st = int(allp.step.iloc[0])
    ntr = int(allp.n_train.median()); nte = int(allp.n_test.median())
    inf = []
    for name, col in [("AUC", "auc"), ("MCC@0.5", "mcc_at_0.5"),
                      ("MCC@heldout-Youden", "mcc_heldout_youden"),
                      ("MCC@oracle", "mcc_oracle")]:
        a = tf[tf.model == "tcn"].sort_values("wid")[col].values
        b = tf[tf.model == "anfis"].sort_values("wid")[col].values
        n = min(len(a), len(b))
        res = full_inference(a[:n], b[:n], list(range(n)), wl, st, ntr, nte)
        inf.append({"metric": name,
                    "naive_p": res["naive_wilcoxon"]["p"], "naive_d": res["naive_wilcoxon"]["cohens_d"],
                    "nonoverlap_p": res["nonoverlap_wilcoxon"]["p"], "nonoverlap_n": res["n_nonoverlap"],
                    "nadeau_bengio_p": res["nadeau_bengio"]["p"]})
    infdf = pd.DataFrame(inf)
    infdf.to_csv(f"{OUT}/inference_corrected_full.csv", index=False)
    print("=== OVERLAP-CORRECTED INFERENCE (primary) ===")
    print(infdf.to_string(index=False)); print()

    print(f"All corrected tables written to {OUT}/")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results_hourly")
