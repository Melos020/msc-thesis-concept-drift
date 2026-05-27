"""DEFINITIVE TCN vs ANFIS concept-drift comparison (hourly, learnable target).

Primary target: vol_regime_h12 -- AUC ~0.71 baseline, so both models reach REAL
skill and concept drift has genuine competence to disrupt. This is what makes
the architectural contrast measurable instead of two flat lines at 0.5.

Protocol (leakage-free, identical for both models):
  * rolling walk-forward over hourly bars; many windows (>=40 per asset)
  * each window: train / pre_holdout / post, chronological
  * scaler fit on TRAIN only; pre_holdout never used for fitting
  * fixed 0.5 threshold (no eval-window tuning)
  * TCN: temporal order intact, 3 seeds averaged at WINDOW level, early stop
  * ANFIS: compact rule base over top-k MI features (train-only selection)
  * window tagged by vol-regime transition train->post for drift decomposition

Metrics: AUC, MCC, balanced acc, accuracy, F1, ECE, DSI, DR.
Inference unit = window. CIs via cluster bootstrap over windows; model
comparisons via paired Wilcoxon over windows.
"""
import os, sys, time, json
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch; torch.set_num_threads(4)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from src.models.tcn import TCN
from src.models.anfis import ANFIS
from corrected_pipeline import make_rolling_windows, full_metrics, corrected_DR
from window_stats import cluster_bootstrap_ci, paired_window_test, selective_prediction
from features_hourly import (load_klines, build_features,
                             target_vol_regime, target_large_move, target_tb_dir)
from src.data.regime import causal_vol_quantile_regime
from run_enriched import (train_tcn_strong, predict_tcn_strong, make_windows_seq)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_hourly")
os.makedirs(OUT, exist_ok=True)


def anfis_compact(Xtr, ytr, Xpre, Xpost, seed=0, k=6, n_mfs=3):
    try:
        mi = mutual_info_classif(Xtr, ytr, random_state=seed)
        sel = np.argsort(mi)[-k:]
    except Exception:
        sel = np.arange(min(k, Xtr.shape[1]))
    an = ANFIS(input_dim=len(sel), n_mfs=n_mfs)
    an.fit_premise(Xtr[:, sel].astype(np.float32), seed=seed)
    an.fit_consequent(Xtr[:, sel].astype(np.float32), ytr.astype(np.float32))
    with torch.no_grad():
        ppre = torch.sigmoid(an(torch.tensor(Xpre[:, sel], dtype=torch.float32))).numpy()
        ppost = torch.sigmoid(an(torch.tensor(Xpost[:, sel], dtype=torch.float32))).numpy()
    return ppre, ppost


def _auc(p, y):
    try:
        return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    except Exception:
        return float("nan")


def eval_window(X, y, w, close=None, seq_len=24, tcn_seeds=(0, 1)):
    Xtr, ytr = X[w.train], y[w.train].astype(int)
    Xpre, ypre = X[w.pre_holdout], y[w.pre_holdout].astype(int)
    Xpost, ypost = X[w.post], y[w.post].astype(int)
    # finite guard: a single inf in a feature window must not crash the run
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xpre = np.nan_to_num(Xpre, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    Xpost = np.nan_to_num(Xpost, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if len(np.unique(ytr)) < 2 or len(np.unique(ypost)) < 2:
        return None
    sc = StandardScaler().fit(Xtr)
    _clip = lambda A: np.nan_to_num(np.clip(A, -1e6, 1e6), nan=0.0, posinf=0.0, neginf=0.0)
    Xtr_s = _clip(sc.transform(Xtr)); Xpre_s = _clip(sc.transform(Xpre)); Xpost_s = _clip(sc.transform(Xpost))
    res = {}

    maj = float(int(ytr.mean() >= 0.5))
    res["majority"] = {"pre": full_metrics(np.full(len(ypre), maj), ypre),
                       "post": full_metrics(np.full(len(ypost), maj), ypost),
                       "auc_post": 0.5, "DR": 0.0}

    # TCN (seeds averaged at window level)
    pre_s, post_s, auc_s, pp0, pp0_pre = [], [], [], None, None
    for s in tcn_seeds:
        m, scl = train_tcn_strong(Xtr, ytr, seq_len=seq_len, seed=s)
        ppre = predict_tcn_strong(m, scl, Xpre, seq_len)
        ppost = predict_tcn_strong(m, scl, Xpost, seq_len)
        if len(ppre) and len(ppost):
            yp = ypre[seq_len:]; yq = ypost[seq_len:]
            pre_s.append(full_metrics(ppre, yp)); post_s.append(full_metrics(ppost, yq))
            auc_s.append(_auc(ppost, yq))
            if pp0 is None: pp0 = (ppost, yq)
            if pp0_pre is None: pp0_pre = (ppre, yp)
    if not post_s:
        return None
    keys = post_s[0].keys(); avg = lambda L, k: float(np.nanmean([x[k] for x in L]))
    mpre = {k: avg(pre_s, k) for k in keys}; mpost = {k: avg(post_s, k) for k in keys}
    res["tcn"] = {"pre": mpre, "post": mpost, "auc_post": float(np.nanmean(auc_s)),
                  "DR": corrected_DR(mpre["accuracy"], mpost["accuracy"]), "_pp": pp0, "_pp_pre": pp0_pre}

    # ANFIS
    try:
        ppre, ppost = anfis_compact(Xtr_s, ytr, Xpre_s, Xpost_s, seed=0, k=6, n_mfs=3)
        mpre = full_metrics(ppre, ypre); mpost = full_metrics(ppost, ypost)
        res["anfis"] = {"pre": mpre, "post": mpost, "auc_post": _auc(ppost, ypost),
                        "DR": corrected_DR(mpre["accuracy"], mpost["accuracy"]), "_pp": (ppost, ypost), "_pp_pre": (ppre, ypre)}
    except Exception as e:
        res["anfis"] = {"error": str(e)}

    # --- classical volatility baselines (close-only, train-fit, leakage-free) ---
    if close is not None:
        from baselines_vol import har_vol_regime, garch_vol_regime
        c_tr, c_pre, c_post = close[w.train], close[w.pre_holdout], close[w.post]
        for nm, fn in [("har", har_vol_regime), ("garch", garch_vol_regime)]:
            try:
                pp_pre, pp_post = fn(c_tr, c_pre, c_post)
                mp = full_metrics(pp_pre, ypre); mq = full_metrics(pp_post, ypost)
                res[nm] = {"pre": mp, "post": mq, "auc_post": _auc(pp_post, ypost),
                           "DR": corrected_DR(mp["accuracy"], mq["accuracy"]),
                           "_pp": (pp_post, ypost), "_pp_pre": (pp_pre, ypre)}
            except Exception as e:
                res[nm] = {"error": str(e)}
    return res


def _find_csv(asset):
    """Locate the hourly CSV for an asset, honoring DATA_DIR and common names."""
    ddir = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw"))
    cands = [f"{ddir}/{asset}USDT_1h.csv", f"{ddir}/{asset}_1h.csv",
             f"{ddir}/{asset}.csv"]
    for c in cands:
        if os.path.exists(c):
            return c
    import glob
    g = glob.glob(f"{ddir}/{asset}*.csv")
    if g:
        return g[0]
    raise FileNotFoundError(f"No hourly CSV for {asset} in {ddir} (tried {cands})")


def run(asset, target="vol_regime", window_len=2400, step=400, downsample=2):
    d = load_klines(_find_csv(asset))
    feats = build_features(d).replace([np.inf, -np.inf], np.nan)
    if target == "vol_regime":
        y = target_vol_regime(d.close)
    elif target == "large_move":
        y = target_large_move(d.close)
    else:
        y = target_tb_dir(d)
    df = feats.join(y.rename("y")).replace([np.inf, -np.inf], np.nan).dropna()
    close_series = d.close.reindex(df.index).values.astype(float)
    if downsample > 1:
        close_series = close_series[::downsample]
        df = df.iloc[::downsample]
    reg = causal_vol_quantile_regime(df["vol"], 4, 60, 60)
    X = df.drop(columns=["vol", "y"]).values.astype(np.float32)
    yv = df["y"].values.astype(int)
    regv = reg.values.astype(int); volv = df["vol"].values.astype(float)
    wins = make_rolling_windows(len(X), regv, volv, window_len=window_len, step=step,
                                train_frac=0.6, pre_frac=0.13)
    print(f"\n### {asset}/{target}: {len(X)} rows (ds={downsample}), {X.shape[1]} feats, {len(wins)} windows")

    rows, agg = [], {m: {k: [] for k in ["auc_post", "mcc_post", "bal_post", "acc_post",
                                         "f1_post", "ece_post", "dsi_post", "acc_pre", "DR"]}
                     for m in ["tcn", "anfis", "har", "garch", "majority"]}
    sel_rows = []
    prob_rows = []
    t0 = time.time()
    for w in wins:
        r = eval_window(X, yv, w, close=close_series)
        if w.wid % 5 == 0: print(f"      win {w.wid}/{len(wins)} t={time.time()-t0:.0f}s", flush=True)
        if r is None:
            continue
        changed = int(w.regime_post != w.regime_train)
        for m in ["tcn", "anfis", "har", "garch", "majority"]:
            if m not in r or "pre" not in r[m]:
                continue
            a = agg[m]
            a["auc_post"].append(r[m]["auc_post"]); a["mcc_post"].append(r[m]["post"]["mcc"])
            a["bal_post"].append(r[m]["post"]["bal_acc"]); a["acc_post"].append(r[m]["post"]["accuracy"])
            a["f1_post"].append(r[m]["post"]["f1"]); a["ece_post"].append(r[m]["post"]["ece"])
            a["dsi_post"].append(r[m]["post"]["dsi"]); a["acc_pre"].append(r[m]["pre"]["accuracy"])
            a["DR"].append(r[m]["DR"])
            rows.append({"wid": w.wid, "model": m, "regime_changed": changed,
                         "auc_post": r[m]["auc_post"], "mcc_post": r[m]["post"]["mcc"],
                         "bal_post": r[m]["post"]["bal_acc"], "acc_post": r[m]["post"]["accuracy"],
                         "ece_post": r[m]["post"]["ece"], "dsi_post": r[m]["post"]["dsi"],
                         "acc_pre": r[m]["pre"]["accuracy"], "DR": r[m]["DR"]})
            if m in ("tcn", "anfis") and "_pp" in r[m]:
                for sp in selective_prediction(r[m]["_pp"][0], r[m]["_pp"][1]):
                    sp.update({"wid": w.wid, "model": m}); sel_rows.append(sp)
            if "_pp" in r[m] and "_pp_pre" in r[m]:
                prob_rows.append({
                    "wid": w.wid, "model": m, "regime_changed": changed,
                    "p_post": list(np.asarray(r[m]["_pp"][0], float)),
                    "y_post": list(np.asarray(r[m]["_pp"][1], int)),
                    "p_pre": list(np.asarray(r[m]["_pp_pre"][0], float)),
                    "y_pre": list(np.asarray(r[m]["_pp_pre"][1], int)),
                    "train_pos_rate": float(yv[w.train].mean()),
                    "n_train": int(len(w.train)), "n_test": int(len(w.post)),
                    "window_len": int(window_len), "step": int(step),
                })
    print(f"    done in {time.time()-t0:.0f}s")

    pd.DataFrame(rows).to_csv(f"{OUT}/{asset}_{target}_windows.csv", index=False)
    pd.DataFrame(sel_rows).to_csv(f"{OUT}/{asset}_{target}_selective.csv", index=False)
    if prob_rows:
        pd.DataFrame(prob_rows).to_pickle(f"{OUT}/{asset}_{target}_window_probs.pkl")

    summ = []
    for m in ["tcn", "anfis", "har", "garch", "majority"]:
        rec = {"model": m, "n_windows": len(agg[m]["mcc_post"])}
        for k in agg[m]:
            vals = agg[m][k]
            if len(vals) < 1 or all(not np.isfinite(v) for v in vals):
                rec[f"{k}_mean"] = float("nan"); rec[f"{k}_lo"] = float("nan"); rec[f"{k}_hi"] = float("nan")
                continue
            ci = cluster_bootstrap_ci(vals)
            rec[f"{k}_mean"] = round(ci["mean"], 4); rec[f"{k}_lo"] = round(ci["lo"], 4); rec[f"{k}_hi"] = round(ci["hi"], 4)
        summ.append(rec)
    pd.DataFrame(summ).to_csv(f"{OUT}/{asset}_{target}_summary.csv", index=False)

    tests = []
    for a, b in [("tcn", "anfis"), ("tcn", "har"), ("tcn", "garch"),
                 ("tcn", "majority"), ("anfis", "majority")]:
        for k in ["auc_post", "mcc_post", "acc_post", "DR", "dsi_post", "ece_post"]:
            va = np.array(agg.get(a, {}).get(k, []), float)
            vb = np.array(agg.get(b, {}).get(k, []), float)
            n = min(len(va), len(vb))
            if n < 3:
                continue
            t = paired_window_test(va[:n], vb[:n])
            t.update({"model_a": a, "model_b": b, "metric": k}); tests.append(t)
    pd.DataFrame(tests).to_csv(f"{OUT}/{asset}_{target}_paired.csv", index=False)

    # drift decomposition
    dfw = pd.DataFrame(rows); dec = []
    for m in ["tcn", "anfis"]:
        sub = dfw[dfw.model == m]
        for lbl, msk in [("same_regime", sub.regime_changed == 0), ("regime_shift", sub.regime_changed == 1)]:
            g = sub[msk]
            if len(g):
                dec.append({"model": m, "transition": lbl, "n": len(g),
                            "auc_post_mean": round(np.nanmean(g.auc_post), 4),
                            "mcc_post_mean": round(np.nanmean(g.mcc_post), 4),
                            "DR_mean": round(np.nanmean(g.DR), 4),
                            "dsi_post_mean": round(np.nanmean(g.dsi_post), 4)})
    pd.DataFrame(dec).to_csv(f"{OUT}/{asset}_{target}_decomp.csv", index=False)
    return summ, tests


if __name__ == "__main__":
    asset = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    target = sys.argv[2] if len(sys.argv) > 2 else "vol_regime"
    run(asset, target)
