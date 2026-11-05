#!/usr/bin/env bash
# Runs the test suite and every experiment script in order, from the
# repository root, exactly as described in README.md.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1_gig
python -m exp.exp2_gb
python -m exp.exp3_gg_gev
python -m exp.exp4_gh
python -m exp.exp5_normal_lognormal
python -m exp.exp6_student_cauchy
python -m exp.exp7_vg
python -m exp.exp8_bs_iv
python -m exp.exp9_bachelier_iv
python -m exp.exp10_verification
