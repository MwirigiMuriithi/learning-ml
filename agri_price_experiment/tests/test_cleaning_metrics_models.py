import numpy as np
import pandas as pd
from agriprice.cleaning import causal_outlier_filter, locf
from agriprice import metrics
from agriprice.features import panel_from_frame, FeatureStore
from agriprice.models import baseline_predictions, fit_predict_ml


def test_outlier_filter_rejects_gross_errors_keeps_level_shift():
    x = np.r_[np.full(10, 50.0), 500.0, np.full(5, 50.0), np.full(8, 90.0)]
    acc, rej = causal_outlier_filter(x)
    assert rej[10] and not rej[11:16].any()
    assert acc[16:].tolist().count(90.0) >= 7          # persistent genuine shift is accepted after one week


def test_outlier_filter_is_causal():
    rng = np.random.default_rng(0); x = 50 * np.exp(np.cumsum(rng.normal(0, .03, 80))); x[40] *= 10
    full, _ = causal_outlier_filter(x); part, _ = causal_outlier_filter(x[:50])
    assert np.array_equal(np.nan_to_num(full[:50]), np.nan_to_num(part))


def test_locf_uses_only_past_and_respects_cap():
    x = np.array([[1.0], [np.nan], [np.nan], [np.nan], [5.0]])
    f, age = locf(x, max_age=2)
    assert f[1, 0] == 1 and f[2, 0] == 1 and np.isnan(f[3, 0]) and f[4, 0] == 5


def test_metrics():
    y, p = np.array([10., 20.]), np.array([12., 18.])
    assert metrics.mae(y, p) == 2 and abs(metrics.rmse(y, p) - 2) < 1e-9
    assert metrics.directional_accuracy(np.array([12., 8.]), np.array([11., 9.]), np.array([10., 10.])) == 1.0
    assert np.isnan(metrics.directional_accuracy(y, y, y))


def test_baselines_and_ml_smoke(panels, cfg):
    fs = FeatureStore(panel_from_frame(panels["realistic"], cfg), cfg, True)
    df = fs.frame(2)
    b = baseline_predictions(df, 2, 105, 130, cfg)
    assert {"naive", "seasonal_naive", "ma4", "ma8"} <= set(b.model)
    assert np.isfinite(b.y_pred).all()
    out = fit_predict_ml("ridge", {"alpha": 100}, "E4", df, 2, 104, 105, 130, cfg, True)
    assert np.isfinite(out.y_pred).all() and (out.y_pred > 0).all()
    # same evaluation rows as baselines
    n = b[b.model == "naive"].shape[0]
    assert len(out) == n
