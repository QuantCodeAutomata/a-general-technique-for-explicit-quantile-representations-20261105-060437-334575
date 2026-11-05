"""Tests for the Normal/log-normal contour quantile construction (exp_5)."""
from __future__ import annotations

import mpmath as mp
import pytest

from src import config
from src import contour_core as cc
from src import normal_lognormal as nl


@pytest.mark.parametrize("mu,sigma", config.NORMAL_TEST_PARAMS)
def test_normal_quantile_matches_reference(mu, sigma):
    """Relative errors below 1e-18 against the SciPy-seeded Newton reference
    (expected_outcomes, exp_5: "errors around 1e-24 or better"; 1e-18 used
    here as a safe but still tight bound for the modest mesh/precision used
    in tests)."""
    contour = nl.build_normal_contour(dps=24, max_cells_per_side=100)
    for p in [0.01, 0.05, 0.5, 0.95, 0.99]:
        q = nl.normal_quantile(contour, p, mu=mu, sigma=sigma, dps=24)
        ref = nl.reference_quantile_normal(mu, sigma, p, dps=30)
        assert cc.relative_error(q, ref) < mp.mpf('1e-18')


@pytest.mark.parametrize("sigma,a", config.LOGNORMAL_TEST_PARAMS)
def test_lognormal_quantile_matches_reference(sigma, a):
    """Relative errors below 1e-15 against the SciPy-seeded Newton reference,
    reusing the same standard-normal contour via the exponential map."""
    contour = nl.build_normal_contour(dps=24, max_cells_per_side=100)
    for p in [0.01, 0.05, 0.5, 0.95, 0.99]:
        q = nl.lognormal_quantile(contour, p, sigma=sigma, a=a, dps=24)
        ref = nl.reference_quantile_lognormal(sigma, a, p, dps=30)
        assert cc.relative_error(q, ref) < mp.mpf('1e-15')


def test_normal_quantile_monotone_in_p():
    """The normal quantile function must be strictly increasing in p (CDF
    monotonicity)."""
    contour = nl.build_normal_contour(dps=24, max_cells_per_side=100)
    ps = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
    qs = [nl.normal_quantile(contour, p, mu=2.0, sigma=0.5, dps=24) for p in ps]
    assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


def test_lognormal_quantile_monotone_in_p():
    """The log-normal quantile function must be strictly increasing in p."""
    contour = nl.build_normal_contour(dps=24, max_cells_per_side=100)
    ps = [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]
    qs = [nl.lognormal_quantile(contour, p, sigma=2.0, a=0.5, dps=24) for p in ps]
    assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


def test_g_normal_matches_scipy_norm_pdf():
    """g_normal(x) for real x must equal scipy.stats.norm.pdf(x) (holomorphic
    continuation sanity check)."""
    from scipy.stats import norm
    for x in [-3.0, -0.5, 0.0, 1.2, 4.0]:
        val = complex(nl.g_normal(mp.mpf(x)))
        ref = norm.pdf(x)
        assert abs(val.real - ref) < 1e-8
        assert abs(val.imag) < 1e-20


def test_same_contour_consistent_across_affine_and_exponential_maps():
    """The SAME standard-normal contour object, fed through the affine map
    (mu=0, sigma=1, i.e. identity) and through the log-normal exponential
    map (sigma=1, a=1, i.e. plain exp), must be consistent with calling
    `cc.residue_quantile` directly on the shared accumulation."""
    contour = nl.build_normal_contour(dps=24, max_cells_per_side=100)
    for p in [0.05, 0.3, 0.7, 0.95]:
        with mp.workdps(24):
            q_direct = cc.residue_quantile(contour.acc, mp.mpf(p))
        q_affine = nl.normal_quantile(contour, p, mu=0.0, sigma=1.0, dps=24, guard_digits=0)
        q_exp = nl.lognormal_quantile(contour, p, sigma=1.0, a=1.0, dps=24, guard_digits=0)
        assert abs(q_affine - q_direct) < mp.mpf('1e-20')
        assert cc.relative_error(q_exp, mp.e ** q_direct) < mp.mpf('1e-15')
