# A general technique for explicit quantile representations

Reproduction of Schadner, "A general technique for explicit quantile
representations" (working paper, 2026).

The paper gives a constructive method that writes the quantile function of a
strictly positive, real-analytic probability density as a contour integral
over a p-independent contour in the complex plane. The contour's height above
the real axis is built once from a phase bound (or, where the phase bound is
not tight enough, a relative-variation bound) on the density's holomorphic
continuation, and the quantile is then recovered for every probability level
p in (0,1) by accumulating the complex CDF along that fixed contour and
evaluating either the logarithmic form (Eq. 1) or the algebraically
equivalent residue form (Eq. 4).

No market or external data of any kind is used: every experiment evaluates
the method on analytically specified working densities (generalized inverse
Gaussian, generalized beta, generalized gamma, generalized extreme value,
generalized hyperbolic, normal/log-normal, Student's t/Cauchy, variance
gamma) and on two option-pricing inversions (Black-Scholes and Bachelier
implied volatility). `data/` is therefore empty of loaders; see
`results/RESULTS.md` for the "Deviations from the paper" and "Not
reproduced" sections.

## Layout

```
src/     contour_core.py (shared quadrature/accumulation engine) plus one
         module per distribution family and per financial application
exp/     one script per experiment, exp1 .. exp10
tests/   pytest unit tests for src/ components
results/ tables, plots and RESULTS.md written by the experiment scripts
data/    empty package; no experiment in this paper uses external or
         synthetic market data
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest
```

## Experiments

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

Each script writes its tables/plots directly to `results/`. See
`results/RESULTS.md` for the paper comparison, deviations and
not-reproduced items.
