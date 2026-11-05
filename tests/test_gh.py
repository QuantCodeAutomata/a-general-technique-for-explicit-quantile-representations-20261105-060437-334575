"""Tests for the generalized hyperbolic (GH) contour quantile (exp_4)."""
import mpmath as mp
import pytest

from src import gh
from src import contour_core as cc


@pytest.mark.parametrize("a,b,d", [(1.0, 1.5, 0.5), (-0.5, 2.0, 1.0), (0.0, 1.0, 0.0)])
def test_gh_residue_quantile_matches_reference(a, b, d):
    """Relative error below 1e-13 at every tested p-level (expected_outcomes,
    exp_4): GH involves two Bessel-K evaluations and a harder reference CDF
    than most other laws in this repository, hence the looser (but still
    tight) bound relative to e.g. GIG's 1e-15/1e-17."""
    contour = gh.build_gh_contour(a, b, d, dps=24, max_cells_per_side=200)
    for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
        q = gh.gh_quantile(contour, p, dps=24)
        ref = gh.reference_quantile_gh(a, b, d, p, dps=30)
        # Symmetric case (d=0) at p=0.5: the true quantile is exactly 0, so
        # both `q` and `ref` are themselves just numerical noise around 0
        # (~1e-26) -- relative error against noise is meaningless, fall back
        # to an absolute check there.
        if abs(ref) < mp.mpf('1e-15'):
            assert abs(q) < mp.mpf('1e-15')
        else:
            assert cc.relative_error(q, ref) < mp.mpf('1e-13')


def test_gh_quantile_monotone_in_p():
    """The quantile function must be strictly increasing in p (CDF monotonicity)."""
    contour = gh.build_gh_contour(1.0, 1.5, 0.5, dps=24, max_cells_per_side=150)
    qs = [gh.gh_quantile(contour, p, dps=24) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


def test_gh_affine_reconstruction():
    """Q~(p) = mu + sigma * Q(p) (methodology, exp_4)."""
    contour = gh.build_gh_contour(1.0, 1.5, 0.5, dps=24, max_cells_per_side=150)
    base = gh.gh_quantile(contour, 0.5, dps=24)
    scaled = gh.gh_quantile_scaled(contour, 0.5, mu=2.0, sigma=3.0, dps=24)
    # The comparison itself must run at (at least) the same working
    # precision the two quantities were computed at, else mpmath rounds the
    # subtraction to the ambient (lower, default) precision and the
    # difference floor is dominated by *that*, not by the computation.
    with mp.workdps(24):
        assert abs(scaled - (mp.mpf(2.0) + mp.mpf(3.0) * base)) < mp.mpf('1e-18')


def test_gh_density_is_holomorphic_positive_real_axis():
    """`g_gh` evaluated on the real axis must match a straightforward,
    real-only reimplementation using `scipy.special.kv` (`g_gh_float`),
    confirming the holomorphic continuation is consistent with the stated
    real density (methodology, exp_4) -- the holomorphic-continuation
    sanity check."""
    for a, b, d in [(1.0, 1.5, 0.5), (-0.5, 2.0, 1.0), (0.0, 1.0, 0.0)]:
        for x in [-2.0, -0.3, 0.0, 0.5, 3.0]:
            val = complex(gh.g_gh(mp.mpf(x), a, b, d))
            ref = gh.g_gh_float(x, a, b, d)
            assert abs(val.real - ref) < 1e-10
            assert abs(val.imag) < 1e-20


def test_gh_density_matches_scipy_genhyperbolic():
    """Independent cross-check of `g_gh_float` (and hence `g_gh` on the real
    axis, verified separately above) against SciPy's `genhyperbolic`, whose
    parametrization (p=tail, a=shape, b=skew) matches this paper's (a, b, d)
    exactly (methodology, exp_4)."""
    from scipy.stats import genhyperbolic
    for a, b, d in [(1.0, 1.5, 0.5), (-0.5, 2.0, 1.0), (0.0, 1.0, 0.0)]:
        for x in [-2.0, -0.3, 0.0, 0.5, 3.0]:
            mine = gh.g_gh_float(x, a, b, d)
            ref = genhyperbolic.pdf(x, a, b, d)
            assert abs(mine - ref) < 1e-8


def test_gh_nig_cross_check_against_scipy_norminvgauss():
    """For the NIG case (a=-0.5), SciPy's `norminvgauss(a_scipy, b_scipy)`
    with `a_scipy=b, b_scipy=d` (standardized, mu=0/sigma=1) matches this
    paper's standardized GH(-0.5, b, d) parametrization exactly -- verified
    numerically against this module's own `g_gh_float` quadrature reference
    before trusting it (methodology, exp_4: "verify this numerically ...
    before trusting it, and only use it as a secondary sanity check")."""
    from scipy.stats import norminvgauss, genhyperbolic
    a, b, d = -0.5, 2.0, 1.0

    # Verify the parametrization mapping itself, independently of our own code.
    for x in [-2.0, -0.3, 0.0, 0.5, 3.0]:
        assert abs(genhyperbolic.pdf(x, a, b, d) - norminvgauss.pdf(x, b, d)) < 1e-10

    contour = gh.build_gh_contour(a, b, d, dps=24, max_cells_per_side=200)
    for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
        q = gh.gh_quantile(contour, p, dps=24)
        q_scipy = norminvgauss.ppf(p, b, d)
        assert abs((float(q) - q_scipy) / q_scipy) < 1e-6 if q_scipy != 0 else abs(float(q)) < 1e-6
