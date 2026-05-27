"""Window-level statistical inference (Phase 2) and selective prediction (Phase 6).

Phase 2 -- the unit of evidence is the WINDOW, never the seed.
    * cluster_bootstrap_ci: resample windows (not seeds) with replacement.
    * paired_window_test: paired comparison across the n independent windows.
    The number of windows is the honest sample size; seeds only reduce
    within-window TCN noise and are averaged away before any test is run.

Phase 6 -- selective prediction on truly held-out windows with calibrated
    probabilities. Compares full / confidence-gated / no-trade-zone policies and
    reports coverage, precision, MCC, balanced accuracy and ECE.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import wilcoxon
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, precision_score
from corrected_pipeline import expected_calibration_error


def cluster_bootstrap_ci(values, n_boot: int = 5000, ci: float = 0.95, seed: int = 0):
    """Percentile bootstrap CI resampling WINDOWS (the independent cluster)."""
    v = np.asarray([x for x in values if not (isinstance(x, float) and np.isnan(x))], float)
    if len(v) < 2:
        return {"mean": float(v.mean()) if len(v) else float("nan"),
                "lo": float("nan"), "hi": float("nan"), "n": int(len(v))}
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    a = (1 - ci) / 2
    return {"mean": float(v.mean()),
            "lo": float(np.quantile(means, a)),
            "hi": float(np.quantile(means, 1 - a)),
            "n": int(len(v))}


def paired_window_test(a, b):
    """Paired Wilcoxon over windows + paired Cohen's d. n = number of windows."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 5 or np.all(a == b):
        d = a - b
        cd = float(d.mean() / d.std(ddof=1)) if (len(d) > 1 and d.std(ddof=1) > 0) else float("nan")
        return {"W": float("nan"), "p": float("nan"), "cohens_d": cd, "n": int(len(a)),
                "mean_a": float(a.mean()) if len(a) else float("nan"),
                "mean_b": float(b.mean()) if len(b) else float("nan")}
    try:
        res = wilcoxon(a, b, alternative="two-sided")
        p = float(res.pvalue); W = float(res.statistic)
    except Exception:
        p, W = float("nan"), float("nan")
    d = a - b
    cd = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
    return {"W": W, "p": p, "cohens_d": cd, "n": int(len(a)),
            "mean_a": float(a.mean()), "mean_b": float(b.mean())}


def selective_prediction(proba_post, y_post, margins=(0.0, 0.05, 0.10, 0.15)):
    """Confidence-gated prediction. For each margin, abstain when
    |p - 0.5| < margin (no-trade zone). margin=0 is full prediction.

    Returns list of dicts with coverage / precision / mcc / bal_acc / ece on the
    RETAINED subset (truly held-out post window, calibrated probabilities).
    """
    proba_post = np.asarray(proba_post, float)
    y_post = np.asarray(y_post, int)
    out = []
    for m in margins:
        keep = np.abs(proba_post - 0.5) >= m
        cov = float(keep.mean()) if len(keep) else 0.0
        if keep.sum() < 5 or len(np.unique(y_post[keep])) < 2:
            out.append({"margin": m, "coverage": cov, "precision": float("nan"),
                        "mcc": float("nan"), "bal_acc": float("nan"),
                        "ece": float("nan"), "n_kept": int(keep.sum())})
            continue
        pk = proba_post[keep]; yk = y_post[keep]
        preds = (pk > 0.5).astype(int)
        try:
            prec = float(precision_score(yk, preds, zero_division=0))
        except Exception:
            prec = float("nan")
        try:
            mcc = float(matthews_corrcoef(yk, preds))
        except Exception:
            mcc = float("nan")
        try:
            bal = float(balanced_accuracy_score(yk, preds))
        except Exception:
            bal = float("nan")
        out.append({"margin": m, "coverage": cov, "precision": prec,
                    "mcc": mcc, "bal_acc": bal,
                    "ece": expected_calibration_error(pk, yk), "n_kept": int(keep.sum())})
    return out
