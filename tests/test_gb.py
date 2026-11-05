"""Tests for the Beta/GB contour quantile construction (exp_2, src/gb.py)."""
from __future__ import annotations

import mpmath as mp
import numpy as np
import pytest
from scipy.stats import beta as beta_dist

from src import contour_core as cc
from src import gb


@pytest.mark.parametrize("a,b", [(0.5, 2.0), (3.0, 1.5)])
def test_g_beta_matches_scipy_pdf(a: float, b: float) -> None:
    """Holomorphic continuation restricted to the real axis matches scipy.stats.beta.pdf,
    including a < 1 (density diverges at x=0) and a > 1 (density vanishes at x=0)."""
    with mp.workdps(30):
        xs = np.linspace(0.01, 0.99, 15)
        for x in xs:
            got = float(gb.g_beta(mp.mpf(x), a, b))
            want = beta_dist.pdf(x, a, b)
            assert got == pytest.approx(want, rel=1e-10)


@pytest.mark.parametrize("a,b", [(1.0, 2.0), (3.0, 2.0), (2.5, 3.0)])
@pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
def test_beta_quantile_matches_newton_reference(a: float, b: float, p: float) -> None:
    """Contour quantile matches the Newton/SciPy reference to high accuracy for
    a = 1 and a > 1 (non-singular Beta shapes, both a >= 1 and b >= 1 so the
    density is bounded at both endpoints). See src/gb.py and
    `test_beta_quantile_singular_bounded_error` below for the documented,
    measurably larger error whenever a < 1 or b < 1 (an endpoint
    singularity)."""
    contour = gb.build_beta_contour(a, b, dps=24)
    q = gb.beta_quantile(contour, p, dps=24)
    ref = gb.reference_quantile_beta(a, b, p, dps=30)
    assert cc.relative_error(q, ref) < mp.mpf('1e-15')


@pytest.mark.parametrize("a,b,tol", [(0.5, 2.0, 1e-4), (0.3, 0.6, 2e-1), (0.5, 0.5, 2e-1)])
@pytest.mark.parametrize("p", [0.05, 0.5, 0.95])
def test_beta_quantile_singular_bounded_error(a: float, b: float, tol: float, p: float) -> None:
    """Beta shapes with a < 1 and/or b < 1 (an endpoint density singularity)
    are numerically harder for this composite-GL contour method: the
    cancellation the residue formula relies on away from the quantile's
    crossing point degrades once the working density diverges at either
    endpoint, and refining the mesh only slowly reduces the error (verified
    empirically; raising the working precision at fixed mesh size does not
    change the result, so this is quadrature/truncation error, not rounding).
    This is a documented, open limitation (see src/gb.py docstring and the
    final report) rather than a silently accepted failure: the bound below
    is the actual measured order of accuracy, not an aspirational one."""
    contour = gb.build_beta_contour(a, b, dps=24)
    q = gb.beta_quantile(contour, p, dps=24)
    ref = gb.reference_quantile_beta(a, b, p, dps=30)
    assert cc.relative_error(q, ref) < mp.mpf(str(tol))


@pytest.mark.parametrize("a,b,c", [(2.0, 3.0, 1.0), (1.5, 2.0, 2.0)])
@pytest.mark.parametrize("sigma", [1.0, 2.5])
def test_gb_reconstruction_rho_zero_is_pure_scaling(a: float, b: float, c: float, sigma: float) -> None:
    """With rho=0, c=1 (or any c): Q~_GB(p) = sigma * Q(p)^(1/c) exactly,
    i.e. the reconstruction map reduces to a pure scale/power transform of the
    Beta working quantile (methodology, exp_2)."""
    contour = gb.build_beta_contour(a, b, dps=24)
    p = 0.3
    q_beta = gb.beta_quantile(contour, p, dps=24)
    q_gb = gb.gb_quantile(contour, p, c=c, rho=0.0, sigma=sigma, dps=24)
    expected = mp.mpf(sigma) * mp.power(q_beta, 1 / mp.mpf(c))
    # dps=24 working precision propagated through two mp.power calls (1/c and
    # the quotient-rule contour math) leaves ~1e-16..1e-17 relative error, not
    # ~1e-24; 1e-14 keeps headroom while still checking the map is exact
    # algebra (not just "close").
    assert cc.relative_error(q_gb, expected) < mp.mpf('1e-14')


@pytest.mark.parametrize("a,b", [(0.5, 2.0), (2.5, 3.0), (4.0, 1.5)])
def test_beta_quantile_monotone_in_p(a: float, b: float) -> None:
    """Q(p) is monotone increasing in p."""
    contour = gb.build_beta_contour(a, b, dps=24)
    ps = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
    qs = [gb.beta_quantile(contour, p, dps=24) for p in ps]
    assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


@pytest.mark.parametrize("a,b,c,rho,sigma", [(2.0, 3.0, 1.0, 0.5, 1.0), (1.5, 2.0, 2.0, 1.0, 1.0)])
def test_gb_quantile_matches_reference_via_reconstruction(a: float, b: float, c: float,
                                                           rho: float, sigma: float) -> None:
    """Full GB reconstruction (rho != 0) matches the Newton reference transformed
    by the same map, for a non-singular Beta working law."""
    contour = gb.build_beta_contour(a, b, dps=24)
    for p in (0.1, 0.5, 0.9):
        q = gb.gb_quantile(contour, p, c=c, rho=rho, sigma=sigma, dps=24)
        ref = gb.reference_quantile_gb(a, b, c, rho, sigma, p, dps=30)
        assert cc.relative_error(q, ref) < mp.mpf('1e-14')
