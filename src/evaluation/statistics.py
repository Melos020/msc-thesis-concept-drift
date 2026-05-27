"""Statistical tests (thesis Ch 4.5).

- Paired Wilcoxon signed-rank test for paired metric comparisons across seeds.
- Cohen's d effect size.
"""
import numpy as np
from scipy.stats import wilcoxon


def paired_wilcoxon(a, b):
    """Returns dict with W stat, p-value, n."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    if len(a) < 3 or np.all(a == b):
        return {'W': float('nan'), 'p': float('nan'), 'n': len(a)}
    try:
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return {'W': float(res.statistic), 'p': float(res.pvalue), 'n': len(a)}
    except Exception:
        return {'W': float('nan'), 'p': float('nan'), 'n': len(a)}


def cohens_d(a, b):
    """Paired Cohen's d (mean diff / std of diffs)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diffs = a - b
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return float('nan')
    return float(diffs.mean() / diffs.std(ddof=1))
