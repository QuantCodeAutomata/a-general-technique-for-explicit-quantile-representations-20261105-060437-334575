"""Tests for the GIG contour quantile (exp_1)."""
import mpmath as mp
import pytest

from src import gig
from src import contour_core as cc


@pytest.mark.parametrize("a,b", [(-0.5, 1.0), (0.5, 2.0), (2.5, 0.5)])
def test_gig_residue_quantile_matches_reference(a, b):
    """Relative errors below 1e-17 for a < 1, a = 1 and a > 1 (expected_outcomes, exp_1)."""
    contour = gig.build_gig_contour(a, b, dps=24, max_cells_per_side=160)
    for p in [0.01, 0.05, 0.5, 0.95, 0.99]:
        q = gig.gig_quantile(contour, p, dps=24)
        ref = gig.reference_quantile_gig(a, b, p, dps=30)
        assert cc.relative_error(q, ref) < mp.mpf('1e-15')


def test_gig_quantile_monotone_in_p():
    """The quantile function must be strictly increasing in p (CDF monotonicity)."""
    contour = gig.build_gig_contour(1.0, 1.0, dps=24, max_cells_per_side=120)
    qs = [gig.gig_quantile(contour, p, dps=24) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


def test_gig_scale_reconstruction():
    """Q~_GIG(p; sigma, a, b) = sigma * Q_GIG(p) (Step 5, exp_1)."""
    contour = gig.build_gig_contour(1.0, 1.0, dps=24, max_cells_per_side=120)
    base = gig.gig_quantile(contour, 0.5, dps=24)
    scaled = gig.gig_quantile_scaled(contour, 0.5, sigma=3.0, dps=24)
    assert abs(scaled - 3 * base) < mp.mpf('1e-18')


def test_gig_density_is_holomorphic_positive_real_axis():
    """g_gig(x) for real x > 0 must equal SciPy's geninvgauss density."""
    from scipy.stats import geninvgauss
    a, b = 1.5, 2.0
    for x in [0.3, 1.0, 3.0]:
        val = complex(gig.g_gig(mp.mpf(x), a, b))
        ref = geninvgauss.pdf(x, a, b)
        assert abs(val.real - ref) < 1e-8
        assert abs(val.imag) < 1e-20
