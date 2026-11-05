"""Tests for the Variance Gamma (VG) half-axis-split contour quantile (exp_7)."""
import mpmath as mp
import pytest

from src import vg
from src import contour_core as cc

PARAMS = [(-0.3, 0.15), (1.5, 0.2), (0.2, -0.1)]

# a<0 (here a=-0.3) has a documented ~1e-6 truncation-floor limitation (see
# "KNOWN LIMITATION" in src/vg.py's module docstring, "Truncation" section):
# g_eps(x) ~ x^(2a) as x->0 for a<0 is more singular than f itself, so the
# mass omitted by the paper's fixed x_lo=1e-14 cutoff is only negligible to
# ~1e-6, not to e^-60; two alternative fixes (smaller per-(a,d) x_lo, and an
# explicit p_left quadrature correction) were tried and empirically made
# accuracy *worse*, so the simple fixed cutoff (the paper's literal recipe)
# is kept and this looser tolerance documents the resulting floor.
TOLERANCES = {-0.3: mp.mpf('5e-6')}
# 1e-11, not 1e-13: with a modest mesh (max_cells_per_side auto-tuned up to
# ~1200 for the a=1.5 case; see build_vg_contour docstring), the a=1.5 case
# achieves ~2e-13 to a few e-12 depending on p (verified directly), i.e.
# consistently below 1e-11 but not uniformly below 1e-13 at every p level;
# VG is numerically harder than every other module in this repository (two
# Bessel-based half-axis quadratures plus a Beta boundary), so 1e-11 is used
# here as the safe, still very tight, bound (task's own guidance: "use
# 1e-13 as a safe bound... document in a comment if you need looser").
DEFAULT_TOL = mp.mpf('1e-11')


def _tol(a):
    return TOLERANCES.get(a, DEFAULT_TOL)


@pytest.fixture(scope="module")
def contours():
    """Build each (a, d) contour once per test session (expensive: Bessel-K
    evaluation at 24+12-digit precision over a few hundred to ~1200 mesh
    points per half-axis)."""
    built = {}
    for a, d in PARAMS:
        built[(a, d)] = vg.build_vg_contour(a, d, dps=24)
    return built


@pytest.mark.parametrize("a,d", PARAMS)
def test_vg_quantile_matches_reference(contours, a, d):
    """Relative error against the Newton-method reference, for a representative
    spread of probability levels (methodology/expected_outcomes, exp_7)."""
    contour = contours[(a, d)]
    tol = _tol(a)
    # p levels chosen to stay reasonably far from p_0 (0.355-0.540 across the
    # three test cases) in all three cases, since relative error necessarily
    # blows up as Q(p) -> 0 near p_0 even for a tiny absolute error.
    for p in [0.05, 0.2, 0.8, 0.95]:
        q = vg.vg_quantile(contour, p, dps=24)
        ref = vg.reference_quantile_vg(a, d, p, dps=30)
        err = cc.relative_error(q, ref)
        assert err < tol, f"a={a}, d={d}, p={p}: relerr={err} >= {tol}"


@pytest.mark.parametrize("a,d", PARAMS)
def test_vg_quantile_at_p0_is_zero(contours, a, d):
    """Q(p_0) = 0 exactly by construction (the two half-axis representations
    meet at zero; methodology, exp_7)."""
    contour = contours[(a, d)]
    q0 = vg.vg_quantile(contour, contour.p0, dps=24)
    assert abs(q0) < mp.mpf('1e-15')


@pytest.mark.parametrize("a,d", PARAMS)
def test_vg_quantile_monotone_across_p0_boundary(contours, a, d):
    """The quantile must be continuous and strictly increasing in p through
    the eps=-1/eps=+1 switch at p=p_0."""
    contour = contours[(a, d)]
    p0 = float(contour.p0)
    eps_small = 0.02
    ps = sorted({max(0.001, p0 - 2 * eps_small), max(0.002, p0 - eps_small),
                 p0, min(0.998, p0 + eps_small), min(0.999, p0 + 2 * eps_small)})
    qs = [vg.vg_quantile(contour, p, dps=24) for p in ps]
    for q1, q2 in zip(qs, qs[1:]):
        assert q2 > q1
    # continuity: the jump across p0 should be of the same order as the
    # neighboring steps, not a discontinuous blow-up
    idx0 = ps.index(p0)
    if 0 < idx0 < len(ps) - 1:
        step_before = qs[idx0] - qs[idx0 - 1]
        step_after = qs[idx0 + 1] - qs[idx0]
        assert step_after < 10 * max(step_before, mp.mpf('1e-6'))


@pytest.mark.parametrize("a,d", PARAMS)
def test_p0_beta_formula_matches_direct_integration(a, d):
    """Cross-check the closed-form Beta-based p_0 against direct numerical
    integration of the real density over (-infinity, 0] (methodology, exp_7:
    "verify p_0 agrees with direct numerical integration... as a sanity
    test"). The a<0 case has the same ~1e-6 near-zero-mass discrepancy
    documented above (direct integration also stops at a fixed -1e-14
    cutoff), so it uses the same looser tolerance."""
    p0_closed = vg.p0_vg(a, d)
    p0_direct = vg.p0_direct_integration(a, d)
    tol = mp.mpf('1e-5') if a < 0 else mp.mpf('1e-10')
    assert abs(p0_closed - p0_direct) < tol


@pytest.mark.parametrize("a,d", PARAMS)
def test_g_eps_holomorphic_continuation_matches_scipy(a, d):
    """g_{+1}(x) for real x>0 must equal the real VG density f(x) for x>0
    (and g_{-1}(x) must equal f(-x)), checked directly against an
    independent SciPy `kv`-based reimplementation (Theorem A.2, exp_7)."""
    import numpy as np
    from scipy.special import kv as scipy_kv, gamma as scipy_gamma

    delta = float(mp.sqrt(1 + d ** 2))
    C = float(np.sqrt(np.pi) * (2 * delta) ** a * scipy_gamma(a + 0.5))

    def f_scipy(x, sign):
        return np.exp(sign * d * x) * x ** a * scipy_kv(a, delta * x) / C

    for x in [0.3, 1.0, 3.0]:
        g_plus = complex(vg.g_vg(mp.mpf(x), a, d, 1))
        g_minus = complex(vg.g_vg(mp.mpf(x), a, d, -1))
        assert abs(g_plus.real - f_scipy(x, 1)) < 1e-8
        assert abs(g_plus.imag) < 1e-20
        assert abs(g_minus.real - f_scipy(x, -1)) < 1e-8
        assert abs(g_minus.imag) < 1e-20
