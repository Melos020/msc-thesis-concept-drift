"""Decision-stability and drift-response metrics (thesis Section 3.5).

This module implements every metric described in the methodology chapter, with
each formula matching the manuscript verbatim. The metrics are:

    accuracy(probs, y)       : standard binary classification accuracy.
    f1_binary(probs, y)      : F1-score with class 1 as positive.

    dsi_norm(probs)          : Decision Stability Index, normalized to [0, 1].
                               Computed as max(0, 1 - 2 * sigma(probs)) over the
                               raw probability stream of one evaluation window
                               (single-window variant; see Section 3.5.2).

    pfr(probs)               : Prediction Flip Rate, fraction of consecutive
                               binary decision flips.

    dce(probs)               : Decision Confidence Entropy, mean per-sample
                               binary Shannon entropy with logarithm base 2.
                               Note that DCE depends only on probs, not on y.

    degradation_ratio(...)   : DR = (acc_pre - acc_post) / acc_pre.
                               Higher values indicate worse drift response;
                               DR = 0 means no degradation; DR = 1 means
                               complete loss; DR < 0 means accuracy improved
                               under drift. This matches the manuscript
                               convention used throughout Chapters 5-7.

    recovery_ratio(...)      : RR = acc_revisit / acc_pre, in [0, 1] for
                               full recovery / catastrophic forgetting.

    memory_effect(...)       : pfr_pre - pfr_post on the PFR metric.
                               Positive => model becomes LESS volatile after
                               drift; negative => model becomes MORE volatile.

All functions are pure NumPy and return Python floats. NaN values are returned
when the input is too short to compute a meaningful estimate.
"""
import numpy as np


# Predictive performance


def accuracy(probs: np.ndarray, y: np.ndarray, thresh: float = 0.5) -> float:
    if len(probs) == 0:
        return float('nan')
    preds = (probs > thresh).astype(int)
    return float((preds == y.astype(int)).mean())


def f1_binary(probs: np.ndarray, y: np.ndarray, thresh: float = 0.5) -> float:
    if len(probs) == 0:
        return float('nan')
    preds = (probs > thresh).astype(int)
    yi = y.astype(int)
    tp = ((preds == 1) & (yi == 1)).sum()
    fp = ((preds == 1) & (yi == 0)).sum()
    fn = ((preds == 0) & (yi == 1)).sum()
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return float(2 * prec * rec / (prec + rec + 1e-9))


# Decision-stability metrics (thesis Section 3.5.2)


def dsi_norm(probs: np.ndarray) -> float:
    """Decision Stability Index (single-window, normalized).

    DSI_norm = max(0, 1 - 2 * sigma(probs))

    where sigma is the population standard deviation of the raw probability
    stream over the evaluation window. The factor of 2 normalizes against the
    maximum standard deviation of any [0, 1]-bounded random variable (which is
    0.5, attained by a Bernoulli(0.5)). The result lies in [0, 1].

    This is the single-window variant of the standard-deviation-of-class-
    proportion stability index. The sub-window construction (with K
    overlapping windows of width w) is an alternative formulation that yields
    similar values; we adopt the single-window form for its mathematical
    simplicity and for the directness of the bounding argument.
    """
    if len(probs) < 2:
        return float('nan')
    return float(max(0.0, 1.0 - 2.0 * np.std(probs)))


def pfr(probs: np.ndarray, thresh: float = 0.5) -> float:
    """Prediction Flip Rate.

    PFR = (1 / (n - 1)) * sum_i 1[ y_hat_i != y_hat_{i+1} ]

    where y_hat_i = 1[ p_i > thresh ]. Bounded in [0, 1]. PFR = 0 means
    the decision stream is constant; PFR = 1 means every consecutive pair
    of decisions disagrees.
    """
    if len(probs) < 2:
        return float('nan')
    preds = (probs > thresh).astype(int)
    flips = (np.abs(np.diff(preds)) > 0).sum()
    return float(flips / (len(preds) - 1))


def dce(probs: np.ndarray, y: np.ndarray = None) -> float:
    """Decision Confidence Entropy.

    DCE = (1 / n) * sum_i H(p_i)
    where H(p) = - p * log2(p) - (1 - p) * log2(1 - p).

    DCE is bounded in [0, 1] (since H of a Bernoulli is bounded by 1 bit).
    DCE close to 0 indicates highly confident predictions (probabilities
    near 0 or 1); DCE close to 1 indicates maximum uncertainty (probabilities
    concentrated near 0.5). DCE depends only on the probability stream and
    does NOT use the labels y; the y argument is retained for API
    compatibility with the other metric functions.
    """
    if len(probs) == 0:
        return float('nan')
    p = np.clip(np.asarray(probs, dtype=float), 1e-9, 1.0 - 1e-9)
    h = -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)
    return float(np.mean(h))



# Derived drift-response measures (thesis Section 3.5.3)


def degradation_ratio(acc_post: float, acc_pre: float) -> float:
    """DR = (acc_pre - acc_post) / acc_pre.

    Higher DR indicates worse drift response. DR = 0 means no degradation;
    DR = 1 means complete loss of accuracy; DR < 0 indicates accuracy improved
    on the post-drift window.

    This convention matches the manuscript text (Section 3.5.3) and all
    tabulated values in Chapters 5-7.
    """
    if acc_pre <= 0 or np.isnan(acc_pre):
        return float('nan')
    return float((acc_pre - acc_post) / acc_pre)


def recovery_ratio(acc_revisit: float, acc_pre: float) -> float:
    """RR = acc_revisit / acc_pre.

    RR close to 1 indicates full recovery to pre-drift competence when the
    training regime reappears (recurring-drift condition). RR << 1 indicates
    catastrophic forgetting.
    """
    if acc_pre <= 0 or np.isnan(acc_pre):
        return float('nan')
    return float(acc_revisit / acc_pre)


def memory_effect(pfr_post: float, pfr_pre: float) -> float:
    """Memory Effect on PFR: pfr_pre - pfr_post.

    Positive => prediction stream becomes LESS volatile after drift (rare).
    Negative => prediction stream becomes MORE volatile after drift (typical).
    Magnitude is bounded in [-1, 1] since PFR is in [0, 1].
    """
    if np.isnan(pfr_pre) or np.isnan(pfr_post):
        return float('nan')
    return float(pfr_pre - pfr_post)


# Convenience aggregator


def all_metrics(probs: np.ndarray, y: np.ndarray) -> dict:
    """Return the five primary window-level metrics in a single dict."""
    return {
        'accuracy': accuracy(probs, y),
        'f1':       f1_binary(probs, y),
        'dsi_norm': dsi_norm(probs),
        'pfr':      pfr(probs),
        'dce':      dce(probs),
    }


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200)
    y = (p > 0.5).astype(int)
    print('--- metrics sanity checks ---')
    print(f'accuracy:         {accuracy(p, y):.4f}')
    print(f'f1:               {f1_binary(p, y):.4f}')
    print(f'dsi_norm:         {dsi_norm(p):.4f}')
    print(f'pfr:              {pfr(p):.4f}')
    print(f'dce:              {dce(p):.4f}    (~1.0 for uniform)')
    p_const = np.full(200, 0.9)
    print(f'dce(const 0.9):   {dce(p_const):.4f}  (should be ~0.469)')
    p_half = np.full(200, 0.5)
    print(f'dce(const 0.5):   {dce(p_half):.4f}  (should be ~1.0)')
    print(f'DR (0.4, 0.5):    {degradation_ratio(0.4, 0.5):.4f}  (should be 0.20)')
    print(f'DR (0.6, 0.5):    {degradation_ratio(0.6, 0.5):.4f}  (should be -0.20)')
