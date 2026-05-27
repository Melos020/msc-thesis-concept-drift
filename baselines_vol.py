"""Classical volatility baselines for the TCN-vs-ANFIS drift study.

ADDS the two strongest reduced-form volatility benchmarks the thesis names in
its State of the Art (Corsi 2009 HAR; Bollerslev 1986 GARCH; Hansen & Lunde
2005 on how hard GARCH(1,1) is to beat) but never actually ran on the
volatility task. Both are wrapped to produce a P(upper-tercile next-12h RV)
probability on EXACTLY the same windows, target, and metric set as the TCN and
ANFIS, so they slot into run_hourly.eval_window without any other change.

Design notes (leakage-free, identical protocol to the learned models):
  * Both baselines are fit on w.train ONLY. The tercile threshold that defines
    the positive class is estimated on TRAIN realized vol only and then applied
    unchanged to pre_holdout and post -- no eval-window information is used.
  * HAR: Corsi's heterogeneous autoregression of log realized vol on its own
    daily/weekly/monthly (here: short/medium/long hourly) averages. We forecast
    next-12h RV, convert to a regime probability via a logistic link fit on the
    train residual spread (train-only), threshold at 0.5 like every other model.
  * GARCH: a GARCH(1,1) on train returns; the 12-step-ahead conditional vol
    forecast is rolled forward and mapped to a regime probability the same way.
  * Neither baseline sees the engineered feature matrix X. They use only the
    causal close series, which is strictly less information than the TCN/ANFIS
    get -- so if a baseline still beats them, the architectural claim is in
    trouble; if it does not, the claim survives a real test.

The functions return (proba_pre, proba_post) aligned to the pre/post label
vectors, exactly like anfis_compact in run_hourly.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from arch import arch_model
from sklearn.linear_model import LogisticRegression



# Shared helpers

def _log_returns(close: np.ndarray) -> np.ndarray:
    c = np.asarray(close, float)
    r = np.zeros_like(c)
    r[1:] = np.log(c[1:] / c[:-1])
    return r


def _realized_vol(returns: np.ndarray, horizon: int = 12) -> np.ndarray:
    """Trailing realized vol over `horizon` bars, causal (uses [t-h+1, t])."""
    s = pd.Series(returns)
    return s.rolling(horizon).std().values


def _future_rv(returns: np.ndarray, horizon: int = 12) -> np.ndarray:
    """Next-`horizon` realized vol, i.e. RV over (t, t+h]. Matches
    features_hourly.target_vol_regime's fut_vol up to the shift convention."""
    s = pd.Series(returns)
    return s.rolling(horizon).std().shift(-horizon).values


def _regime_prob_from_forecast(fc_train, rv_train, fc_eval, thr):
    """Map a continuous RV forecast to P(upper tercile) via a train-only
    logistic calibrator. fc_* are forecasts; rv_train is the realized future RV
    on train used to build the binary target; thr is the TRAIN tercile cut."""
    y_train = (rv_train > thr).astype(int)
    m = (~np.isnan(fc_train)) & (~np.isnan(rv_train))
    if m.sum() < 20 or len(np.unique(y_train[m])) < 2:
        # fall back to a hard monotone map: prob rises through the threshold
        def _hard(fc):
            fc = np.asarray(fc, float)
            out = np.full(len(fc), 0.5)
            ok = ~np.isnan(fc)
            out[ok] = (fc[ok] > thr).astype(float)
            return out
        return _hard(fc_eval)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(fc_train[m].reshape(-1, 1), y_train[m])
    fe = np.asarray(fc_eval, float)
    out = np.full(len(fe), 0.5)
    ok = ~np.isnan(fe)
    out[ok] = lr.predict_proba(fe[ok].reshape(-1, 1))[:, 1]
    return out



# HAR (Corsi 2009), adapted to hourly bars and a 12-bar horizon

def har_vol_regime(close_train, close_pre, close_post, horizon=12,
                   short=12, med=72, long=360):
    """Corsi-style HAR forecast of next-12h RV, turned into a regime
    probability. short/med/long are the averaging windows (hourly analogue of
    Corsi's daily/weekly/monthly). Fit on train only."""
    rtr = _log_returns(close_train)
    rv = _realized_vol(rtr, horizon)                     
    fut = _future_rv(rtr, horizon)                       

    s = pd.Series(rv)
    har_s = s.rolling(short).mean().values
    har_m = s.rolling(med).mean().values
    har_l = s.rolling(long).mean().values

    Xtr = np.column_stack([har_s, har_m, har_l])
    m = np.all(np.isfinite(Xtr), axis=1) & np.isfinite(fut)
    if m.sum() < 50:
 
        return (np.full(len(close_pre), 0.5), np.full(len(close_post), 0.5))

    eps = 1e-12
    ytr = np.log(fut[m] + eps)
    Atr = np.column_stack([np.ones(m.sum()), np.log(Xtr[m] + eps)])
    beta, *_ = np.linalg.lstsq(Atr, ytr, rcond=None)

    def _forecast(close_block, hist_close):
        """Forecast next-12h RV for each row of close_block, using HAR averages
        computed on [hist_close ++ close_block] so the rolling windows are
        causal and warm. Returns RV forecast aligned to close_block rows."""
        full = np.concatenate([hist_close, close_block])
        r = _log_returns(full)
        rvf = _realized_vol(r, horizon)
        sf = pd.Series(rvf)
        hs = sf.rolling(short).mean().values
        hm = sf.rolling(med).mean().values
        hl = sf.rolling(long).mean().values
        A = np.column_stack([np.ones(len(full)), np.log(hs + eps),
                             np.log(hm + eps), np.log(hl + eps)])
        logfc = A @ beta
        fc = np.exp(logfc) - eps
        return fc[len(hist_close):]                        

    fc_tr = (np.column_stack([np.ones(m.sum()), np.log(Xtr[m] + eps)]) @ beta)
    fc_tr = np.exp(fc_tr) - eps
    rv_tr_target = fut[m]
    thr = np.nanquantile(rv_tr_target, 0.667)

    fc_pre = _forecast(np.asarray(close_pre, float), np.asarray(close_train, float))
    fc_post = _forecast(np.asarray(close_post, float), np.asarray(close_train, float))

    p_pre = _regime_prob_from_forecast(fc_tr, rv_tr_target, fc_pre, thr)
    p_post = _regime_prob_from_forecast(fc_tr, rv_tr_target, fc_post, thr)
    return p_pre, p_post



# GARCH(1,1) (Bollerslev 1986)

def garch_vol_regime(close_train, close_pre, close_post, horizon=12):
    """GARCH(1,1) conditional-vol forecast of next-12h RV -> regime probability.
    Fit on train returns only; forecast rolled forward over pre and post."""
    rtr = _log_returns(close_train)[1:] * 100.0          
    if len(rtr) < 100 or np.allclose(rtr.std(), 0):
        return (np.full(len(close_pre), 0.5), np.full(len(close_post), 0.5))

    try:
        am = arch_model(rtr, mean="Zero", vol="GARCH", p=1, q=1, dist="t")
        fit = am.fit(disp="off", show_warning=False)
    except Exception:
        return (np.full(len(close_pre), 0.5), np.full(len(close_post), 0.5))

    # Build train forecast series: h-step-ahead conditional vol, aggregated to a
    # 12-bar RV-equivalent (sqrt of summed conditional variances).
    cv_train = fit.conditional_volatility / 100.0         # back to return scale
    # next-12h RV proxy on train = sqrt(sum of next-12 conditional variances).
    # Approximate with rolling forward sum of cv^2 (causal target built on train).
    cv2 = pd.Series(cv_train ** 2)
    fut_rv_tr = np.sqrt(cv2.shift(-horizon).rolling(horizon).sum().values)
    fc_tr = np.sqrt(cv2.rolling(horizon).sum().values)    # in-sample fitted level

    def _roll_forecast(close_block, hist_close):
        """Refit-free rolling: feed each new return, read the model's one-step
        conditional vol via fixed params, aggregate to a 12-bar level.
        Returns EXACTLY len(close_block) values, aligned to the block rows."""
        nb = len(close_block)
        full = np.concatenate([hist_close, close_block])
        try:
            r = _log_returns(full) * 100.0          # same length as `full`
            am2 = arch_model(r[1:], mean="Zero", vol="GARCH", p=1, q=1, dist="t")
            filt = am2.fix(fit.params)
            cv = np.concatenate([[np.nan], filt.conditional_volatility]) / 100.0
        except Exception:
            cv = pd.Series(_log_returns(full)).rolling(horizon).std().values
        cv = np.asarray(cv, float)
        cv2b = pd.Series(cv ** 2)
        lvl = np.sqrt(cv2b.rolling(horizon).sum().values)  
        out = lvl[-nb:]                                       
        return out

    thr = np.nanquantile(fut_rv_tr[np.isfinite(fut_rv_tr)], 0.667)
    fc_pre = _roll_forecast(np.asarray(close_pre, float), np.asarray(close_train, float))
    fc_post = _roll_forecast(np.asarray(close_post, float), np.asarray(close_train, float))

    p_pre = _regime_prob_from_forecast(fc_tr, fut_rv_tr, fc_pre, thr)
    p_post = _regime_prob_from_forecast(fc_tr, fut_rv_tr, fc_post, thr)
    return p_pre, p_post
