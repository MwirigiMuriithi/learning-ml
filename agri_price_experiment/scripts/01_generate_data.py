#!/usr/bin/env python
"""Stage 1-2: generate the clean latent world, apply the three data-quality scenarios, save everything."""
import sys, json, hashlib, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from agriprice.config import load_config, data_dir, SCENARIOS
from agriprice.generator import generate_clean_world, event_table
from agriprice.degrade import degrade


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config"); ap.add_argument("--seed", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    seed = cfg["experiment"]["seed"] if a.seed is None else a.seed
    out = data_dir()
    world = generate_clean_world(cfg, seed)
    world.to_csv(out / "clean_world_audit.csv", index=False)
    ev = event_table(cfg); ev.to_csv(out / "events_schedule.csv", index=False)
    manifest = dict(seed=seed, config=cfg["_path"], config_sha256=sha(cfg["_path"]), files={})
    anns = []
    for sc in SCENARIOS:
        panel, ann = degrade(world, cfg, sc, seed)
        f = out / f"synthetic_panel_{sc}.csv"
        panel.to_csv(f, index=False)
        try:
            panel.to_parquet(out / f"synthetic_panel_{sc}.parquet", index=False)
        except Exception:
            pass
        anns.append(ann)
        manifest["files"][f.name] = dict(rows=len(panel), sha256=sha(f))
        print(f"{sc:10s} rows={len(panel)}  missing price={panel.wholesale_price_kes_kg.isna().mean():.3%}  "
              f"flags={panel.data_quality_flag.value_counts().to_dict()}")
    pd.concat(anns, ignore_index=True).to_csv(out / "data_quality_annotations.csv", index=False)
    for n in ["events_schedule.csv", "data_quality_annotations.csv", "clean_world_audit.csv"]:
        manifest["files"][n] = dict(sha256=sha(out / n))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out)

if __name__ == "__main__":
    main()
