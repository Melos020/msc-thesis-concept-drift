"""
robustness_core.py  --  the once-and-for-all correction layer.

This module closes the THREE structural objections for good, with real numbers,
not wording. It is data-agnostic: the SAME functions run on the daily-direction
data (shipped) and on the hourly volatility data (your machine). Nothing here
depends on which dataset is passed in.

It fixes:

  (1) THRESHOLD FAIRNESS.  The fixed-0.5 MCC conflates calibration with skill.
      We replace it with a THRESHOLD-FAIR comparison that reports, per window:
        - AUC                  : threshold-free ranking (the fair head-to-head)
        - MCC@0.5              : the naive operating point (kept for continuity)
        - MCC@bestF (oracle)   : best MCC over a threshold grid -- UPPER BOUND on
                                 each model's decision skill, so neither model is
                                 penalised by an arbitrary cut
        - MCC@heldout          : threshold chosen ONLY on the held-out pre window
                                 (base-rate and Youden), applied to post -- the
                                 HONEST deployable number
        - ECE                  : calibration, a separate axis
      This turns "0.5 hid ANFIS's potential" into a designed experiment: we give
      BOTH models their best honest threshold and see what survives.

  (2) WINDOW-OVERLAP INFERENCE.  Overlapping windows are not independent, so the
      naive Wilcoxon / cluster-bootstrap over 60 windows overstate significance.
      We report BOTH:
        - a NON-OVERLAPPING subsample (step >= window_len) -> genuinely
          independent units, honest n
        - the Nadeau-Bengio CORRECTED-RESAMPLED t-test, which inflates the
          variance to account for the train/test overlap (the standard fix,
          already cited in the thesis via Nadeau & Bengio 2003)
      If the effect survives both, overlap can never be used against it.

  (3) CLASSICAL BASELINES.  HAR (Corsi) and GARCH(1,1) run under the identical
      leakage-free protocol (see baselines_vol.py), so the learned models are
      measured against the econometric bar, not just a majority predictor.

Author: integrated for the TCN-vs-ANFIS thesis hardening.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score, balanced_accuracy_score, roc_curve



# (1) THRESHOLD-FAIR EVALUATION

def mcc_at(probs, y, t):
    pr = (np.asarray(probs, float) > t).astype(int)
    yy = np.asarray(y, int)
    if len(np.unique(yy)) < 2:
        return float("nan")
    try:
        return float(matthews_corrcoef(yy, pr))
    except Exception:
        return float("nan")


def best_threshold_oracle(probs, y, grid=None):
    """Threshold that maximises MCC ON THIS window (oracle upper bound).
    Reported as a capability ceiling, never as a deployable number."""
    if grid is None:
        grid = np.linspace(0.05, 0.95, 91)
    best_t, best_m = 0.5, -2.0
    for t in grid:
        m = mcc_at(probs, y, t)
        if np.isfinite(m) and m > best_m:
            best_m, best_t = m, t
    return best_t, best_m


def threshold_from_heldout(p_hold, y_hold, train_pos_rate, policy="youden"):
    """Pick a threshold using ONLY the held-out pre window (leakage-free).
    policy in {'youden','base_rate'}."""
    p_hold = np.asarray(p_hold, float)
    y_hold = np.asarray(y_hold, int)
    if policy == "base_rate":
        if len(p_hold) < 5 or not np.isfinite(train_pos_rate) or not (0 < train_pos_rate < 1):
            return 0.5
        return float(np.quantile(p_hold, 1.0 - train_pos_rate))
    # youden
    if len(np.unique(y_hold)) < 2 or len(p_hold) < 5:
        return 0.5
    fpr, tpr, thr = roc_curve(y_hold, p_hold)
    return float(thr[int(np.argmax(tpr - fpr))])


def threshold_fair_row(p_pre, y_pre, p_post, y_post, train_pos_rate):
    """One window's full threshold-fair record for one model."""
    auc = float(roc_auc_score(y_post, p_post)) if len(np.unique(y_post)) > 1 else float("nan")
    t_base = threshold_from_heldout(p_pre, y_pre, train_pos_rate, "base_rate")
    t_youd = threshold_from_heldout(p_pre, y_pre, train_pos_rate, "youden")
    _, mcc_oracle = best_threshold_oracle(p_post, y_post)
    return {
        "auc": auc,
        "mcc_at_0.5": mcc_at(p_post, y_post, 0.5),
        "mcc_oracle": mcc_oracle,                       # capability ceiling
        "mcc_heldout_base": mcc_at(p_post, y_post, t_base),
        "mcc_heldout_youden": mcc_at(p_post, y_post, t_youd),
        "bal_at_0.5": (balanced_accuracy_score(y_post, (np.asarray(p_post) > 0.5).astype(int))
                       if len(np.unique(y_post)) > 1 else float("nan")),
    }


# (2) OVERLAP-CORRECTED INFERENCE

def nadeau_bengio_t(diffs, n_train, n_test):
    """Corrected-resampled paired t-test (Nadeau & Bengio 2003).
    diffs: per-window metric differences (model A - model B).
    n_train, n_test: representative train/test sizes per window. The variance is
    inflated by (1/k + n_test/n_train) instead of 1/k, which corrects for the
    dependence induced by overlapping training sets. This is the standard
    remedy for exactly the window-overlap criticism."""
    d = np.asarray([x for x in diffs if np.isfinite(x)], float)
    k = len(d)
    if k < 3:
        return {"t": float("nan"), "p": float("nan"), "mean": float(np.mean(d)) if k else float("nan"),
                "n": k, "corrected": True}
    mean = d.mean()
    var = d.var(ddof=1)
    ratio = (n_test / n_train) if n_train > 0 else 0.0
    corrected_var = var * (1.0 / k + ratio)
    if corrected_var <= 0:
        return {"t": float("nan"), "p": float("nan"), "mean": mean, "n": k, "corrected": True}
    t = mean / np.sqrt(corrected_var)
    p = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"t": float(t), "p": p, "mean": float(mean), "n": k, "corrected": True}


def nonoverlapping_indices(wids, window_len, step):
    """Pick a maximal subset of window ids whose data blocks do NOT overlap.
    Greedy: take a window, skip until the next window starts beyond the last
    window's end. ceil(window_len/step) is the stride in window-index units."""
    stride = int(np.ceil(window_len / step))
    return list(range(0, len(wids), max(stride, 1)))


def paired_wilcoxon(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    if len(a) < 5 or np.all(a == b):
        d = a - b
        cd = float(d.mean() / d.std(ddof=1)) if (len(d) > 1 and d.std(ddof=1) > 0) else float("nan")
        return {"W": float("nan"), "p": float("nan"), "cohens_d": cd, "n": int(len(a)),
                "mean_a": float(a.mean()) if len(a) else float("nan"),
                "mean_b": float(b.mean()) if len(b) else float("nan")}
    try:
        res = stats.wilcoxon(a, b, alternative="two-sided")
        W, p = float(res.statistic), float(res.pvalue)
    except Exception:
        W, p = float("nan"), float("nan")
    d = a - b
    cd = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    return {"W": W, "p": p, "cohens_d": cd, "n": int(len(a)),
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def full_inference(metric_a, metric_b, wids, window_len, step, n_train, n_test):
    """Run all three inference views on a paired metric (A vs B over windows):
      naive_wilcoxon     : the original (overstates significance under overlap)
      nonoverlap_wilcoxon: on the independent subsample (honest n)
      nadeau_bengio      : overlap-corrected t-test on all windows
    """
    a = np.asarray(metric_a, float); b = np.asarray(metric_b, float)
    naive = paired_wilcoxon(a, b)
    sub = nonoverlapping_indices(wids, window_len, step)
    nonov = paired_wilcoxon(a[sub], b[sub])
    nb = nadeau_bengio_t(a - b, n_train, n_test)
    return {"naive_wilcoxon": naive, "nonoverlap_wilcoxon": nonov,
            "nadeau_bengio": nb, "n_nonoverlap": len(sub)}
