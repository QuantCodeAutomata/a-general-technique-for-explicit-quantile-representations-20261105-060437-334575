# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this implements

A reproduction of the constructive method of Schadner, "A general technique
for explicit quantile representations": the quantile function of a strictly
positive, real-analytic density is written as a contour integral over a
p-independent contour in the complex plane (logarithmic form Eq. 1, residue
form Eq. 4), with the contour height derived once from a phase bound or
relative-variation bound on the density's holomorphic continuation.

Ten experiments are implemented, one per distribution family or financial
application: GIG (exp_1), generalized beta (exp_2), generalized gamma / GEV
(exp_3), generalized hyperbolic (exp_4), normal / log-normal (exp_5),
Student's t / Cauchy (exp_6), variance gamma (exp_7), Black-Scholes implied
volatility (exp_8), Bachelier implied volatility (exp_9), and a cross-law
numerical verification over twelve working laws and eight probability levels
(exp_10).

No market or other external data is used anywhere in this repository: every
experiment evaluates the method on analytically specified densities and
option-pricing functions, so `data/` holds only an empty package.

## Layout

```
src/      contour_core.py (shared mpmath contour/quadrature engine, mesh
          construction, CDF accumulation, logarithmic and residue quantile
          forms, Newton reference solver) plus one module per family:
          gig.py, gb.py, gg_gev.py, gh.py, normal_lognormal.py,
          student_cauchy.py, vg.py, bs_iv.py, bachelier_iv.py, verification.py,
          config.py (all paper-derived parameters as module-level variables)
exp/      one script per experiment, exp1_gig.py .. exp10_verification.py;
          each calls only the src/ functions and writes its outputs to
          results/
tests/    pytest unit tests for src/ components (synthetic/analytic inputs
          only; no market_data marker is used because no experiment loads
          market data)
results/  tables (CSV), plots (PNG) and RESULTS.md written by the experiment
          scripts; rerunning an experiment overwrites its files
data/     empty package; no experiment in this paper uses external or
          synthetic market data
```

## Commands

Setup:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Tests:

```bash
python -m pytest -m "not market_data"
```

(No test in this repository is marked `market_data`; the flag is kept for
consistency with the shared testing convention.)

Experiments (run from the repository root):

```bash
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
```

`reproduce.sh` runs the full test suite followed by every experiment script
in this order.

## Conventions

* All experiment parameters (orders, concentrations, scales, probability
  grids, mpmath working precisions) live only in `src/config.py` as
  module-level `UPPER_CASE` variables, each with a one-line comment stating
  its meaning, unit and source section/equation in the paper (or "not
  specified in the paper").
* `src/contour_core.py` is the single shared implementation of the mesh
  construction, complex-CDF accumulation, and the two quantile forms
  (logarithmic Eq. 1, residue Eq. 4); every distribution module builds its
  contour on top of it instead of re-implementing quadrature.
* Reference quantiles use SciPy (double precision) as a Newton starting
  point and mpmath Newton iteration at `NEWTON_REFERENCE_DPS` for the
  high-precision comparison value.
* No market or API data is read anywhere in this repository; there is
  consequently no `.env` or `.env.example`.
* After a rerun, the experiment scripts overwrite the tables and plots in
  `results/`; the narrative numbers and verdicts in `results/RESULTS.md`
  must then be checked against the freshly written files and updated by
  hand, they are not regenerated automatically.
