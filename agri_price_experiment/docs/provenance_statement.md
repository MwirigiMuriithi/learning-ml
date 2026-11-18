# Provenance and Intended-Use Statement

**What this is.** `data/synthetic_panel_*.csv` is a *fully synthetic* weekly wholesale-price panel (5 markets x 4 crops x 156 weeks). It was produced by
`scripts/01_generate_data.py` from `config/experiment_config.yaml` with seed 42. No real price, market, weather or trade record
was used, copied, or calibrated against; place names (Nairobi, Nakuru, Eldoret, Kisumu, Mombasa) and KES units are used only to make the structure readable.

**What it is for.** Comparing forecasting methods under *controlled, known* conditions: known causal drivers, documented shocks, adjustable data quality
and a hidden ground truth (`latent_price_kes_kg`) that lets us separate model error from measurement noise.

**What it must not be used for.**
- Any statement about real prices, real markets or real policy; any trading, procurement, credit or aid decision.
- Claims that a method that wins here will win on real data. Real markets contain mechanisms, regime changes and data problems that this generator does not.
- Estimating real-world elasticities: every coefficient is an assumption written in the configuration file.

**Known simplifications.** One national weather/fuel structure with market scaling; stationary price level around a deterministic seasonal pattern (no multi-year inflation or trend);
prices respond to arrivals but arrivals do not respond to past prices; no data revisions (a "corrected" flag appears only in pipeline-cleaned data); errors are gross
(x10/x0.1/x2.5/x0.4/zero) rather than subtle; only one realisation of the world (seed) per run, so conclusions should be confirmed across seeds.

**Reproducibility.** `python scripts/01_generate_data.py` regenerates all files bit-for-bit (SHA-256 in `data/manifest.json`).

File checksums: 
- `synthetic_panel_clean.csv`: `9b3d8ee2a620430f...`
- `synthetic_panel_realistic.csv`: `3a9903cce5f1b919...`
- `synthetic_panel_stressed.csv`: `c1838888362fd12e...`
- `events_schedule.csv`: `027a290bc70400ea...`
- `data_quality_annotations.csv`: `5febeb47558002a9...`
- `clean_world_audit.csv`: `33fec58e3185012a...`
