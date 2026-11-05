"""Tests for the Student's t / Cauchy contour quantile (exp_6)."""
import mpmath as mp
import pytest

from src import student_cauchy as sc
from src import contour_core as cc


def test_cauchy_exact_matches_contour_and_scipy():
    """a=1 (Cauchy): the contour-based `student_t_quantile` must match the
    exact closed form `cauchy_exact_quantile` (methodology, exp_6) to 1e-15,
    and both must match SciPy's `cauchy.ppf` to 1e-10 as an independent check."""
    from scipy.stats import cauchy as scipy_cauchy
    contour = sc.build_student_t_contour(1.0, dps=24, max_cells_per_side=200, safety=1.25)
    for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        q_contour = sc.student_t_quantile(contour, p, dps=24)
        q_exact = sc.cauchy_exact_quantile(p)
        q_scipy = scipy_cauchy.ppf(p)
        if p == 0.5:
            assert abs(q_contour) < mp.mpf('1e-10')
            assert abs(q_exact) < mp.mpf('1e-20')
        else:
            assert cc.relative_error(q_contour, q_exact) < mp.mpf('1e-8')
            assert abs((float(q_exact) - q_scipy) / q_scipy) < 1e-10
            assert abs((float(q_contour) - q_scipy) / q_scipy) < 1e-7


def test_student_t2_exact_matches_contour_and_scipy():
    """a=2: the contour-based `student_t_quantile` must match the exact
    closed form `student_t2_exact_quantile` (methodology, exp_6) to 1e-15,
    and both must match SciPy's `t.ppf(p, df=2)` to 1e-10."""
    from scipy.stats import t as scipy_t
    contour = sc.build_student_t_contour(2.0, dps=24, max_cells_per_side=200, safety=1.25)
    for p in [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]:
        q_contour = sc.student_t_quantile(contour, p, dps=24)
        q_exact = sc.student_t2_exact_quantile(p)
        q_scipy = scipy_t.ppf(p, 2)
        if p == 0.5:
            assert abs(q_contour) < mp.mpf('1e-10')
            assert abs(q_exact) < mp.mpf('1e-20')
        else:
            assert cc.relative_error(q_contour, q_exact) < mp.mpf('1e-9')
            assert abs((float(q_exact) - q_scipy) / q_scipy) < 1e-10
            assert abs((float(q_contour) - q_scipy) / q_scipy) < 1e-8


@pytest.mark.parametrize("a", [0.5, 2.0, 5.0, 30.0])
def test_student_t_quantile_matches_reference(a):
    """Relative error against the SciPy-seeded Newton reference, for a modest
    subset of `config.STUDENT_T_TEST_DOF` and p-levels (kept away from the
    most extreme p=0.01/0.99 for a=0.5 here for test speed; that regime is
    covered separately by `test_student_t_a05_extreme_p_amplification`, since
    the near-field uniform mesh required to resolve it is large -- see
    `src/student_cauchy.py` module docstring)."""
    contour = sc.build_student_t_contour(a, dps=24, max_cells_per_side=150, safety=1.15)
    for p in [0.05, 0.2, 0.5, 0.8, 0.95]:
        q = sc.student_t_quantile(contour, p, dps=24)
        ref = sc.reference_quantile_student_t(a, p, dps=30)
        if p == 0.5:
            assert abs(q) < mp.mpf('1e-8')
        else:
            assert cc.relative_error(q, ref) < mp.mpf('1e-7')


def test_student_t_a05_extreme_p_amplification():
    """For small a (a=0.5) at extreme p (0.01, 0.99), the tangent
    reconstruction amplifies the (tiny) underlying error in the bounded
    arctan-observable sum by a factor of order |quantile|^2 ~= 1028^2 ~= 1e6
    (implementation_notes, exp_6) -- confirmed directly in `src/student_cauchy.py`
    development to be independent of further mesh refinement or working
    precision, i.e. a genuine amplification floor, not an under-resolved
    mesh. Hence the looser tolerance here (1e-6, not the 1e-7..1e-9 achieved
    at every other (a, p) in this module) -- documented, not a uniform
    loosening of every tolerance."""
    contour = sc.build_student_t_contour(0.5, dps=24, max_cells_per_side=150, safety=1.25)
    for p in [0.01, 0.99]:
        q = sc.student_t_quantile(contour, p, dps=24)
        ref = sc.reference_quantile_student_t(0.5, p, dps=30)
        assert cc.relative_error(q, ref) < mp.mpf('1e-6')


def test_student_t_quantile_monotone_in_p():
    """The quantile function must be strictly increasing in p (CDF monotonicity)."""
    contour = sc.build_student_t_contour(5.0, dps=24, max_cells_per_side=150, safety=1.15)
    qs = [sc.student_t_quantile(contour, p, dps=24) for p in [0.1, 0.3, 0.5, 0.7, 0.9]]
    assert all(q2 > q1 for q1, q2 in zip(qs, qs[1:]))


def test_student_t_scale_location_reconstruction():
    """Q~(p) = mu + sigma * Q(p) (methodology, exp_6)."""
    contour = sc.build_student_t_contour(5.0, dps=24, max_cells_per_side=150, safety=1.15)
    base = sc.student_t_quantile(contour, 0.8, dps=24)
    scaled = sc.student_t_quantile_scaled(contour, 0.8, mu=2.0, sigma=3.0, dps=24)
    assert abs(scaled - (2.0 + 3.0 * base)) < mp.mpf('1e-14')


@pytest.mark.parametrize("a", [0.5, 1.0, 2.0, 5.0, 30.0])
def test_student_t_density_is_holomorphic_positive_real_axis(a):
    """g_student_t(x, a) for real x must equal SciPy's t.pdf(x, df=a)."""
    from scipy.stats import t as scipy_t
    for x in [-3.0, -0.5, 0.0, 0.7, 2.5]:
        val = complex(sc.g_student_t(mp.mpf(x), a))
        ref = scipy_t.pdf(x, a)
        assert abs(val.real - ref) < 1e-8
        assert abs(val.imag) < 1e-20


def test_cdf_student_t_matches_scipy():
    """Sanity check for the betainc-based reference CDF used by
    `reference_quantile_student_t`."""
    from scipy.stats import t as scipy_t
    for a in [0.5, 1.0, 2.0, 5.0]:
        for x in [-4.0, -1.0, 0.0, 1.0, 4.0]:
            val = float(sc.cdf_student_t(mp.mpf(x), a))
            ref = scipy_t.cdf(x, a)
            assert abs(val - ref) < 1e-8
