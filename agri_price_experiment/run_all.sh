#!/usr/bin/env bash
# Reproduce the whole experiment end-to-end (about 30-40 minutes on one CPU core).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"
python scripts/01_generate_data.py            # Stage 1-2: synthetic world + 3 data-quality scenarios
python -m pytest tests -q                     # leakage / reproducibility / acceptance tests
python scripts/02_audit_and_eda.py            # acceptance checks, EDA report + figures
python scripts/03_tune.py                     # Stage 3: rolling-origin tuning (validation info only)
python scripts/04_run_experiments.py --part all   # Stage 4-6: models, scenarios, ablation, generalisation, intervals
python scripts/05_analyse.py                  # Stage 7: tables, statistics, figures, hypothesis verdicts
python scripts/06_make_docs.py                # data dictionary, generator spec, provenance, data contract
python scripts/07_build_reports.py            # FINAL_REPORT.md, ERROR_ANALYSIS.md
echo "done - see reports/FINAL_REPORT.md"
