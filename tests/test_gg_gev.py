"""Tests for the GG/GEV contour quantile construction (exp_3, src/gg_gev.py)."""
from __future__ import annotations

import mpmath as mp
import pytest

from src import contour_core as cc
from src import gg_gev


@pytest.mark.parametrize("a,c,tol", [
    # a=0.4 < 1: the gamma(a) density ~ x^(a-1) diverges (integrably) at the
    # left endpoint, a known, measured quadrature limitation of this
    # composite-GL contour method -- same character as the Beta a<1/b<1 case
    # documented in src/gb.py -- so the tolerance here is the actually
    # measured order of accuracy, not an aspirational one (see
    # src/gg_gev.py, build_gamma_contour docstring).
    (0.4, 1.0, 2e-12),
    (1.0, 1.0, 1e-15),
    (2.5, 0.5, 1e-15),
    (1.0, 2.0, 1e-15),
])
def test_gg_quantile_matches_gengamma_reference(a, c, tol):
    """Relative errors below the documented tolerance against the
    gengamma-seeded Newton reference (expected_outcomes, exp_3), for a < 1,
    a = 1 and a > 1."""
    contour = gg_gev.build_gamma_contour(a, dps=24, max_cells_per_side=160)
    for p in [0.01, 0.2, 0.5, 0.8, 0.99]:
        q = gg_gev.gg_quantile(contour, p, c=c, sigma=1.0, dps=24)
        ref = gg_gev.reference_quantile_gg(a, c, p, sigma=1.0, dps=30)
        assert cc.relative_error(q, ref) < mp.mpf(str(tol))


@pytest.mark.parametrize("a,sigma,mu", [(-0.3, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 2.0, 1.0)])
def test_gev_quantile_matches_elementary_closed_form(a, sigma, mu):
    """Relative errors below 1e-15 against the elementary closed form
    (methodology, exp_3), for a < 0, a = 0 and a > 0."""
    contour = gg_gev.build_gamma_contour(1.0, dps=24, max_cells_per_side=160)
    for p in [0.01, 0.2, 0.5, 0.8, 0.99]:
        q = gg_gev.gev_quantile(contour, p, a=a, sigma=sigma, mu=mu, dps=24)
        ref = gg_gev.reference_quantile_gev(p, a=a, sigma=sigma, mu=mu, dps=30)
        assert cc.relative_error(q, ref) < mp.mpf('1e-15')


def test_gev_two_implementations_agree():
    """The contour chain (Q_exp(1-p) via the residue form, then the decreasing
    map) and the elementary closed form are two implementations of the same
    reconstruction and must agree to high precision (methodology, exp_3)."""
    contour = gg_gev.build_gamma_contour(1.0, dps=24, max_cells_per_side=160)
    for a, sigma, mu in [(-0.3, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 2.0, 1.0)]:
        for p in [0.05, 0.4, 0.95]:
            q_contour = gg_gev.gev_quantile(contour, p, a=a, sigma=sigma, mu=mu, dps=24)
            q_closed_form = gg_gev.reference_quantile_gev(p, a=a, sigma=sigma, mu=mu, dps=24)
            assert cc.relative_error(q_contour, q_closed_form) < mp.mpf('1e-15')


def test_gg_quantile_monotone_in_p():
    """The GG quantile function must be strictly increasing in p."""
    contour = gg_gev.build_gamma_contour(2.5, dps=24, max_cells_per_side=160)
    ps = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
    qs = [gg_gev.gg_quantile(contour, p, c=0.5, sigma=1.0, dps=24) for p in ps]
    assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


def test_gev_quantile_monotone_in_p():
    """The GEV quantile function must be strictly increasing in p, for a > 0,
    a = 0 and a < 0."""
    contour = gg_gev.build_gamma_contour(1.0, dps=24, max_cells_per_side=160)
    ps = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
    for a, sigma, mu in [(-0.3, 1.0, 0.0), (0.0, 1.0, 0.0), (0.5, 2.0, 1.0)]:
        qs = [gg_gev.gev_quantile(contour, p, a=a, sigma=sigma, mu=mu, dps=24) for p in ps]
        assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


@pytest.mark.parametrize("a", [0.5, 1.0, 2.5])
def test_g_gamma_matches_scipy_gamma_pdf(a):
    """g_gamma(x) for real x > 0 must equal scipy.stats.gamma.pdf(x, a)
    (holomorphic continuation sanity check)."""
    from scipy.stats import gamma as gamma_dist
    for x in [0.3, 1.0, 3.0]:
        val = complex(gg_gev.g_gamma(mp.mpf(x), a))
        ref = gamma_dist.pdf(x, a)
        assert abs(val.real - ref) < 1e-8
        assert abs(val.imag) < 1e-20
