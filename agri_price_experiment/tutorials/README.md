# Tutorials - learn the experiment step by step

Eleven notebooks that explain, in detail and in order, how the forecasting experiment works - and, just as
importantly, how such an experiment can quietly go wrong. They are written for someone who wants to *understand*
the project, not just reproduce it.

> **Everything is synthetic.** Place names and prices are for readability only. Nothing here describes real markets.

## How the tutorials relate to the real project

| | The pipeline (`src/`, `scripts/`) | The tutorials (`tutorials/`) |
|---|---|---|
| Purpose | Produce the experiment's results | Explain *why* each step exists |
| Code | **The source of truth** | A simplified, standalone re-implementation (`notebooks/tutorial_utils.py`) |
| Data | Generates it | **Reads** the shipped `data/` and `results/` - never writes to them |
| If they disagree | The pipeline is right | The tutorial gets fixed |

Consequences worth knowing:

* Numbers you compute in a notebook can differ a little from the final report (fewer features, default settings, shortcuts).
  Where practical a notebook prints the pipeline's number next to its own. **Only the pipeline's numbers are the experiment's results.**
* The notebooks are shipped **already executed** (outputs included), so you can read them without running anything.
* `tutorial_utils.py` does not import `src/agriprice`. Each notebook ends with a "Where this lives in the pipeline" pointer so you can read the real implementation next.
* Notebooks 08-09 refit small models; notebook 10 only reads the shipped prediction files.

## Reading order

| # | Notebook | The question it answers | Key "aha" |
|---|---|---|---|
| 00 | `00_start_here` | Orientation, glossary, setup check | - |
| 01 | `01_the_forecasting_problem` | What exactly is predicted, and how is it scored? | Skill vs naive; why scale-free metrics |
| 02 | `02_building_the_synthetic_world` | Where do the prices come from? | Build a toy generator; each ingredient has a visible effect |
| 03 | `03_data_quality` | What does messy data look like; how to clean *without* cheating? | A causal outlier filter, its two failure modes, and the cost of not cleaning |
| 04 | `04_eda_with_discipline` | What may exploration decide - and what must it not? | Exploration can leak the test period into your choices |
| 05 | `05_leakage` | How can a model look brilliant for the wrong reason? | **Deliberate leaks**, and the truncation / perturbation tests that expose them |
| 06 | `06_validation_splits` | Why chronological splits, purging, rolling origins? | Random CV flatters flexible models at long horizons |
| 07 | `07_baselines` | Why is "tomorrow equals today" so hard to beat? | Seasonal naive wins for exactly one crop |
| 08 | `08_features_and_models` | What do the models see; what does each feature group add? | Predict the change, not the level; cross-market information |
| 09 | `09_tuning_and_statistics` | How to tune honestly and tell real differences from luck? | Edge-of-grid check; paired block bootstrap |
| 10 | `10_stress_tests_and_errors` | When and why do the models fail? What changes on real data? | The edge largely disappears after the regime shift |

Each notebook ends with **exercises**. Do them - the best way to understand a pipeline is to break it on purpose.

## Running them

```bash
pip install -r requirements.txt                 # the project's requirements
pip install -r tutorials/requirements-tutorials.txt
jupyter lab tutorials/notebooks                 # or open them in VS Code
```

Keep the `tutorials/` folder **inside** the project folder (the helper finds the project root by looking for
`config/experiment_config.yaml`). Running all notebooks takes a few minutes on one CPU core (the slowest are 08 and 09, about 30-40 s each).

## Keeping them honest (drift)

* `python tutorials/check_tutorials.py` executes every notebook in a temporary copy and reports any that fail. Add `--save` to refresh the shipped outputs.
* Passing means "runs without error". It does **not** check that the *written explanation* still matches the printed numbers.
  The explanations were written after running the notebooks and checked against their outputs for the shipped run (seed 42, pipeline version 1.0.0).
  If you change the seed or the configuration, re-read the text against the outputs.
* `tutorial_utils.py` carries a "last checked against pipeline version" stamp. When the pipeline changes, update the stamp after re-checking.

## Things the tutorials deliberately do that the pipeline never does

* Commit leakage on purpose (notebook 05), including cleaning with the audit labels.
* Use a *simpler* feature set (25 features instead of 55), default model settings and a one-line causal cleaner.
* Print intermediate objects that the pipeline keeps internal (single-origin tables, raw filters, fold diagrams).
