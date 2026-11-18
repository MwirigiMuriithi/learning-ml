import numpy as np
from agriprice.generator import generate_clean_world, event_table
from agriprice.degrade import degrade


def test_shape_and_keys(world, cfg):
    assert len(world) == 5 * 4 * cfg["experiment"]["n_weeks"]
    assert not world.duplicated(["market_id", "crop_id", "week_index"]).any()
    assert (world["latent_price_kes_kg"] > 0).all()


def test_reproducible_and_seed_sensitive(world, cfg):
    again = generate_clean_world(cfg)
    assert again.equals(world)
    other = generate_clean_world(cfg, seed=cfg["experiment"]["seed"] + 1)
    assert not np.allclose(other["latent_price_kes_kg"], world["latent_price_kes_kg"])


def test_degradation_reproducible(world, cfg):
    a, _ = degrade(world, cfg, "realistic"); b, _ = degrade(world, cfg, "realistic")
    assert a.equals(b)


def test_clean_scenario_has_no_missing(panels):
    p = panels["clean"]
    assert p["wholesale_price_kes_kg"].notna().all()
    assert (p["data_quality_flag"].isin(["clean", "synthetic_shock"])).all()


def test_flag_precedence(panels):
    p = panels["realistic"]
    assert (p.loc[p.wholesale_price_kes_kg.isna(), "data_quality_flag"] == "missing").all()
    assert (p["data_quality_flag"] == "suspect").sum() > 0


def test_network_weights_sum_to_one(cfg):
    for m, srcs in cfg["network"].items():
        assert abs(sum(srcs.values()) - 1) < 1e-9


def test_event_table_has_test_novelty(cfg):
    ev = event_table(cfg)
    assert ev["new_in_test"].any() and (ev.loc[ev.new_in_test, "partition"] == "test").all()
