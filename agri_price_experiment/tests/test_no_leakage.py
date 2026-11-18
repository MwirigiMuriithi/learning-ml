import numpy as np
from agriprice.features import panel_from_frame, FeatureStore, feature_columns
from agriprice.splits import partitions, rolling_folds
from agriprice.models import training_frame, evaluation_frame
from agriprice.validate import leakage_truncation_test


def test_truncation_invariance_handled(panels, cfg):
    ok, diff = leakage_truncation_test(panels["stressed"], cfg, True)
    assert ok and diff == 0.0


def test_truncation_invariance_raw(panels, cfg):
    ok, diff = leakage_truncation_test(panels["stressed"], cfg, False)
    assert ok


def test_features_do_not_use_future_values(panels, cfg):
    """Corrupting every observation after week T0 must not change features of origins <= T0."""
    df = panels["realistic"].copy()
    T0 = 90
    base = FeatureStore(panel_from_frame(df, cfg), cfg, True).frame(4, with_targets=False)
    d2 = df.copy(); late = d2.week_index > T0
    for c in ["wholesale_price_kes_kg", "market_arrivals_tonnes", "rainfall_mm", "fuel_cost_index"]:
        d2.loc[late, c] = d2.loc[late, c] * 7 + 1
    alt = FeatureStore(panel_from_frame(d2, cfg), cfg, True).frame(4, with_targets=False)
    cols = feature_columns("E4", base.columns)[0]
    a = base[base.origin_week <= T0][cols].to_numpy(float); b = alt[alt.origin_week <= T0][cols].to_numpy(float)
    assert np.array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))


def test_late_indicator_not_visible_at_origin(panels, cfg):
    """A record delivered one week late must be invisible at its own origin week (lag 0)."""
    df = panels["stressed"].copy()
    row = df[(df.report_delay_weeks == 1) & df.market_arrivals_tonnes.notna() & (df.week_index == 80)].iloc[0]
    sel = (df.market_id == row.market_id) & (df.crop_id == row.crop_id) & (df.week_index == 80)
    f1 = FeatureStore(panel_from_frame(df, cfg), cfg, False).frame(1, with_targets=False)
    d2 = df.copy(); d2.loc[sel, "market_arrivals_tonnes"] = 1e6
    f2 = FeatureStore(panel_from_frame(d2, cfg), cfg, False).frame(1, with_targets=False)
    m = (f1.market_id == row.market_id) & (f1.crop_id == row.crop_id) & (f1.origin_week == 80)
    assert f1.loc[m, "arr_l0"].equals(f2.loc[m, "arr_l0"])          # not visible at week 80 ...
    m81 = (f1.market_id == row.market_id) & (f1.crop_id == row.crop_id) & (f1.origin_week == 81)
    assert not f1.loc[m81, "arr_l1"].equals(f2.loc[m81, "arr_l1"])  # ... visible one week later


def test_training_rows_are_purged(panels, cfg):
    fs = FeatureStore(panel_from_frame(panels["realistic"], cfg), cfg, True)
    P = partitions(cfg)
    for h in cfg["experiment"]["horizons"]:
        df = fs.frame(h)
        tr = training_frame(df, h, P["train"][1], cfg, True)
        assert tr["target_week"].max() <= P["train"][1]
        ev = evaluation_frame(df, *P["validation"])
        assert ev["origin_week"].min() > tr["origin_week"].max()
        tr2 = training_frame(df, h, P["validation"][1], cfg, True)
        assert tr2["target_week"].max() <= P["validation"][1]


def test_rolling_folds_are_expanding_and_ordered(cfg):
    f = rolling_folds(cfg)
    assert len(f) == 5 and f[-1]["val_end"] == cfg["experiment"]["splits"]["val_end"]
    assert all(f[i]["train_end"] < f[i + 1]["train_end"] for i in range(len(f) - 1))
    assert all(x["val_start"] == x["train_end"] + 1 for x in f)
