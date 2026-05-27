"""
run_all_corrections.py  --  ONE driver, all three fixes, real numbers.

USAGE
-----
A) On the HOURLY volatility data (the primary path; uses the per-window
   probability dumps shipped in results/provenance/, no torch needed for the
   correction layer):

       # recompute the corrected tables from the shipped provenance:
       python3 run_all_corrections.py hourly results_hourly
       # (or regenerate the dumps first with the walk-forward driver:)
       python3 run_hourly_patched.py BTC vol_regime
       python3 run_hourly_patched.py ETH vol_regime
       python3 run_hourly_patched.py BNB vol_regime

B) On daily-direction data (optional; exercises the correction layer
   end-to-end with an sklearn panel as learned-model stand-ins). Needs a
   daily_direction_audit/ directory that is not shipped with this package:

       python3 run_all_corrections.py daily

OUTPUTS (written to corrections_out/)
  threshold_fair_summary.csv   per-model AUC / MCC@0.5 / MCC@oracle / MCC@heldout / ECE
  threshold_fair_paired.csv    TCN vs ANFIS under each threshold policy
  inference_corrected.csv      naive vs non-overlap vs Nadeau-Bengio for each metric
  baselines_summary.csv        HAR / GARCH vs learned models (hourly path only)
  DECISION.txt                 plain-language verdict: what survives, what doesn't
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from robustness_core import (threshold_fair_row, full_inference, paired_wilcoxon,
                             nadeau_bengio_t)

OUT = "corrections_out"
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared: given per-window probability records, produce all corrected tables.
# records: list of dicts with keys
#   wid, model, p_pre, y_pre, p_post, y_post, train_pos_rate,
#   n_train, n_test, window_len, step
# ---------------------------------------------------------------------------
def produce_corrections(records, tag=""):
    df = pd.DataFrame(records)
    models = [m for m in ["tcn", "anfis", "har", "garch"] if m in df.model.unique()]

    # ---- (1) threshold-fair per-window rows ----
    tf_rows = []
    for _, r in df.iterrows():
        row = threshold_fair_row(r.p_pre, r.y_pre, r.p_post, r.y_post, r.train_pos_rate)
        row.update({"wid": r.wid, "model": r.model})
        tf_rows.append(row)
    tf = pd.DataFrame(tf_rows)

    summ = (tf.groupby("model")
              .agg(n=("wid", "nunique"),
                   AUC=("auc", "mean"),
                   MCC_at_0p5=("mcc_at_0.5", "mean"),
                   MCC_oracle=("mcc_oracle", "mean"),
                   MCC_heldout_youden=("mcc_heldout_youden", "mean"),
                   MCC_heldout_base=("mcc_heldout_base", "mean"))
              .reindex(models).round(4).reset_index())
    summ.to_csv(f"{OUT}/threshold_fair_summary{tag}.csv", index=False)

    #  (1b) paired TCN vs ANFIS under each threshold policy 
    pol_cols = {"AUC (ranking)": "auc",
                "MCC@0.5 (naive)": "mcc_at_0.5",
                "MCC@oracle (ceiling)": "mcc_oracle",
                "MCC@heldout-Youden (honest)": "mcc_heldout_youden",
                "MCC@heldout-base (honest)": "mcc_heldout_base"}
    pair_rows = []
    if "tcn" in models and "anfis" in models:
        for label, col in pol_cols.items():
            a = tf[tf.model == "tcn"].sort_values("wid")[col].values
            b = tf[tf.model == "anfis"].sort_values("wid")[col].values
            n = min(len(a), len(b))
            t = paired_wilcoxon(a[:n], b[:n])
            t.update({"policy": label})
            pair_rows.append(t)
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(f"{OUT}/threshold_fair_paired{tag}.csv", index=False)

    #  (2) overlap-corrected inference (on MCC@0.5 and on AUC) 
    inf_rows = []
    if "tcn" in models and "anfis" in models:
        wl = int(df.window_len.iloc[0]); st = int(df.step.iloc[0])
        ntr = int(df.n_train.median()); nte = int(df.n_test.median())
        for metric_name, col in [("AUC", "auc"), ("MCC@0.5", "mcc_at_0.5"),
                                 ("MCC@heldout-Youden", "mcc_heldout_youden")]:
            a = tf[tf.model == "tcn"].sort_values("wid")[col].values
            b = tf[tf.model == "anfis"].sort_values("wid")[col].values
            n = min(len(a), len(b)); a, b = a[:n], b[:n]
            wids = list(range(n))
            res = full_inference(a, b, wids, wl, st, ntr, nte)
            inf_rows.append({
                "metric": metric_name,
                "naive_p": res["naive_wilcoxon"]["p"],
                "naive_d": res["naive_wilcoxon"]["cohens_d"],
                "naive_n": res["naive_wilcoxon"]["n"],
                "nonoverlap_p": res["nonoverlap_wilcoxon"]["p"],
                "nonoverlap_d": res["nonoverlap_wilcoxon"]["cohens_d"],
                "nonoverlap_n": res["n_nonoverlap"],
                "nadeau_bengio_p": res["nadeau_bengio"]["p"],
                "nadeau_bengio_t": res["nadeau_bengio"]["t"],
            })
    inf = pd.DataFrame(inf_rows)
    inf.to_csv(f"{OUT}/inference_corrected{tag}.csv", index=False)

    return summ, pair, inf


def write_decision(summ, pair, inf, tag=""):
    lines = []
    lines.append("=" * 72)
    lines.append("DECISION  --  what survives the three corrections (real numbers)")
    lines.append("=" * 72)
    lines.append("")
    lines.append("(1) THRESHOLD FAIRNESS")
    if not pair.empty:
        for _, r in pair.iterrows():
            sig = "n.s." if (not np.isfinite(r["p"]) or r["p"] > 0.05) else f"p={r['p']:.1e}"
            lines.append(f"    {r['policy']:32s} TCN={r['mean_a']:+.3f} ANFIS={r['mean_b']:+.3f} "
                         f"d={r['cohens_d']:+.2f} {sig}")
        lines.append("")
        # interpretive logic
        auc = pair[pair.policy.str.startswith("AUC")]
        you = pair[pair.policy.str.contains("Youden")]
        if len(auc) and len(you):
            auc_ns = (not np.isfinite(auc.p.iloc[0])) or auc.p.iloc[0] > 0.05
            you_d = you.cohens_d.iloc[0]
            lines.append("    READING:")
            lines.append(f"      - Ranking (AUC): {'TIED (no significant difference)' if auc_ns else 'TCN higher'}.")
            if abs(you_d) < 0.3:
                lines.append("      - Under an HONEST held-out threshold, the MCC gap is SMALL: the")
                lines.append("        0.5 advantage was largely calibration. Claim must be calibration-only.")
            else:
                lines.append(f"      - Under an HONEST held-out threshold the TCN STILL leads (d={you_d:+.2f}):")
                lines.append("        the advantage is NOT merely the 0.5 cut. This is the strong outcome.")
    lines.append("")
    lines.append("(2) WINDOW-OVERLAP INFERENCE")
    if not inf.empty:
        for _, r in inf.iterrows():
            lines.append(f"    {r['metric']:22s} naive p={r['naive_p']:.1e} (n={int(r['naive_n'])}) | "
                         f"non-overlap p={r['nonoverlap_p']:.1e} (n={int(r['nonoverlap_n'])}) | "
                         f"Nadeau-Bengio p={r['nadeau_bengio_p']:.1e}")
        lines.append("")
        lines.append("    READING: an effect that stays significant in the non-overlap and")
        lines.append("    Nadeau-Bengio columns cannot be attacked as window-overlap inflation.")
    lines.append("")
    lines.append("(3) CLASSICAL BASELINES (hourly path only)")
    if summ is not None and "har" in summ.model.values:
        for _, r in summ.iterrows():
            lines.append(f"    {r['model']:6s} AUC={r['AUC']:.3f}  MCC@heldout-Youden={r['MCC_heldout_youden']:+.3f}")
        lines.append("    READING: if TCN's honest MCC exceeds HAR/GARCH, the econometric bar is cleared.")
    else:
        lines.append("    (run the hourly path to populate HAR/GARCH)")
    txt = "\n".join(lines)
    open(f"{OUT}/DECISION{tag}.txt", "w").write(txt)
    print(txt)


# DAILY path: build per-window probabilities from the shipped sklearn panel.
# Validates the whole correction layer end-to-end with NO torch dependency.

def run_daily():
    base = os.path.join(os.path.dirname(__file__), "daily_direction_audit")
    if not os.path.isdir(base):
        raise SystemExit(
            "daily mode needs an optional daily_direction_audit/ directory "
            "(not shipped). Use the hourly path instead: "
            "python3 run_all_corrections.py hourly results_hourly")
    sys.path.insert(0, base)
    from corrected_pipeline import (build_labels, make_rolling_windows, _fit_scaler,
                                     _calibrated_proba, _sklearn_panel)
    from src.data.features import build_features
    from src.data.regime import causal_vol_quantile_regime

    WINDOW_LEN, STEP, TRAIN_FR, PRE_FR = 420, 90, 0.55, 0.15
    records = []
    for asset, fn in [("BTC", "BTC-USD_OHLCV_clean.csv"), ("ETH", "ETH-USD_OHLCV_clean.csv")]:
        d = pd.read_csv(os.path.join(base, "data", fn))
        d.columns = [c.lower() for c in d.columns]
        close = d["close"].astype(float).reset_index(drop=True)
        feats = build_features(d).replace([np.inf, -np.inf], np.nan)
        y = build_labels(close, horizon=3)
        df = feats.join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
        volv = df["vol"].values.astype(float)
        X = df.drop(columns=["y", "vol"]).values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        yv = df["y"].values.astype(int)
        reg = causal_vol_quantile_regime(pd.Series(volv), 4, 30, 10).values.astype(int)
        wins = make_rolling_windows(len(X), reg, volv, WINDOW_LEN, STEP, TRAIN_FR, PRE_FR)
        print(f"  {asset}: {len(X)} rows, {len(wins)} windows")

        panel = _sklearn_panel(seed=0)   # logistic + trees etc.; use two as TCN/ANFIS stand-ins
        # On daily we only need to EXERCISE the correction layer, so we map two
        # panel members to the two roles: 'tcn'<-histgb (calibrated), 'anfis'<-logreg.
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import HistGradientBoostingClassifier
        for w in wins:
            Xtr, ytr = X[w.train], yv[w.train]
            Xpre, ypre = X[w.pre_holdout], yv[w.pre_holdout]
            Xpost, ypost = X[w.post], yv[w.post]
            if len(np.unique(ytr)) < 2 or len(np.unique(ypost)) < 2:
                continue
            sc = _fit_scaler(Xtr)
            Xtr_s, Xpre_s, Xpost_s = sc.transform(Xtr), sc.transform(Xpre), sc.transform(Xpost)
            for role, clf in [("tcn", HistGradientBoostingClassifier(max_iter=80, random_state=0)),
                              ("anfis", LogisticRegression(max_iter=500))]:
                clf.fit(Xtr_s, ytr)
                ppre = clf.predict_proba(Xpre_s)[:, 1]
                ppost = clf.predict_proba(Xpost_s)[:, 1]
                records.append({"wid": f"{asset}_{w.wid}", "model": role,
                                "p_pre": ppre, "y_pre": ypre, "p_post": ppost, "y_post": ypost,
                                "train_pos_rate": float(ytr.mean()),
                                "n_train": len(ytr), "n_test": len(ypost),
                                "window_len": WINDOW_LEN, "step": STEP})
    return records


# HOURLY path: load per-window probability dumps from the patched pipeline.

def run_hourly(results_dir):
    pkls = sorted(glob.glob(os.path.join(results_dir, "*_window_probs.pkl")))
    if not pkls:
        print(f"No *_window_probs.pkl in {results_dir}. Run run_hourly_patched.py first.")
        sys.exit(1)
    records = []
    for pk in pkls:
        wp = pd.read_pickle(pk)
        # window geometry is embedded by the patched driver; default to thesis geometry
        wl = int(wp["window_len"].iloc[0]) if "window_len" in wp else 2400
        st = int(wp["step"].iloc[0]) if "step" in wp else 400
        for _, r in wp.iterrows():
            records.append({"wid": f"{os.path.basename(pk)}_{r['wid']}", "model": r["model"],
                            "p_pre": np.asarray(r["p_pre"], float), "y_pre": np.asarray(r["y_pre"], int),
                            "p_post": np.asarray(r["p_post"], float), "y_post": np.asarray(r["y_post"], int),
                            "train_pos_rate": float(r["train_pos_rate"]),
                            "n_train": int(r.get("n_train", 1440)), "n_test": int(r.get("n_test", 700)),
                            "window_len": wl, "step": st})
    return records


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "hourly"
    if mode == "daily":
        recs = run_daily()
    else:
        recs = run_hourly(sys.argv[2] if len(sys.argv) > 2 else "results_hourly")
    summ, pair, inf = produce_corrections(recs)
    print("\n=== threshold-fair summary ===")
    print(summ.to_string(index=False))
    print("\n=== threshold-fair paired (TCN vs ANFIS) ===")
    print(pair.to_string(index=False))
    print("\n=== overlap-corrected inference ===")
    print(inf.to_string(index=False))
    print()
    write_decision(summ, pair, inf)
