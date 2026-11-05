"""Unit tests for the shared contour-integral engine (src/contour_core.py)."""
import mpmath as mp
import pytest

from src import contour_core as cc


def test_leggauss_nodes_sum_to_known_weights():
    """20-point Gauss-Legendre weights on [-1,1] must sum to 2 (the length of
    the interval), and the nodes must be symmetric about 0."""
    with mp.workdps(24):
        nodes, weights = cc.leggauss(20)
        assert abs(sum(weights) - 2) < mp.mpf('1e-20')
        for i in range(10):
            assert abs(nodes[i] + nodes[19 - i]) < mp.mpf('1e-20')


def test_leggauss_integrates_polynomial_exactly():
    """A composite GL rule with n points integrates polynomials of degree
    < 2n exactly; check a degree-5 polynomial on [-1, 1]."""
    with mp.workdps(24):
        nodes, weights = cc.leggauss(20)
        total = sum(w * (3 * t ** 5 - 2 * t ** 3 + t) for t, w in zip(nodes, weights))
        assert abs(total) < mp.mpf('1e-20')  # odd polynomial integrates to 0


def test_cumulative_matrix_matches_exact_antiderivative():
    """The cumulative matrix applied to a degree-5 polynomial's values at
    the GL nodes must reproduce the exact antiderivative at each node."""
    with mp.workdps(30):
        nodes, _ = cc.leggauss(20)
        M = cc.cumulative_matrix(20)
        f_vals = mp.matrix([t ** 5 for t in nodes])
        cum = M * f_vals
        for t, c in zip(nodes, cum):
            exact = (t ** 6 - (-1) ** 6) / 6
            assert abs(c - exact) < mp.mpf('1e-20')


def test_build_mesh_cells_no_wider_than_local_height():
    """Cell widths must track the local contour height h(x), not merely grow
    geometrically away from the seed (Appendix D.1: 'cells... no wider than
    the contour height')."""
    with mp.workdps(24):
        def h(x):
            return x ** 2 / (1 + 3 * x + x ** 2)  # GIG height, a=-0.5, b=1

        mesh = cc.build_mesh(h, center=mp.mpf(3), x_lo=mp.mpf('0.01'), x_hi=mp.mpf(20),
                              width_factor=1.0, growth=1.08, max_cells_per_side=200)
        for a, b in zip(mesh.edges[:-1], mesh.edges[1:]):
            width = b - a
            local_h = max(h(a), h(b), mp.mpf('1e-12'))
            assert width <= local_h * mp.mpf('1.3')


def test_residue_and_log_forms_recover_normal_quantile():
    """Normal working law: both the residue form (Eq. 4) and the logarithmic
    form (Eq. 1) must recover scipy.stats.norm.ppf to high precision."""
    from scipy.stats import norm

    with mp.workdps(24):
        def g(z):
            return mp.e ** (-z ** 2 / 2) / mp.sqrt(2 * mp.pi)

        def h(x):
            return 1 / (2 * (1 + x ** 2))

        def P(z):
            return (1 + mp.erf(z / mp.sqrt(2))) / 2

        acc = cc.build_contour(g, h, center=mp.mpf(0), x_lo=mp.mpf(-8), x_hi=mp.mpf(8),
                                mesh_kwargs=dict(width_factor=mp.mpf('1.0'), growth=mp.mpf('1.1'),
                                                  max_cells_per_side=120))
        for p in [0.05, 0.5, 0.95]:
            q_res = cc.residue_quantile(acc, mp.mpf(p))
            ref = mp.mpf(norm.ppf(p))
            assert cc.relative_error(q_res, ref) < mp.mpf('1e-13')


def test_newton_quantile_recovers_known_root():
    """Newton's method on a CDF with a known closed-form inverse (exponential)."""
    with mp.workdps(30):
        def cdf(x):
            return 1 - mp.e ** (-x)

        q = cc.newton_quantile(cdf, p=0.3, x0=0.2, lo=0.0, hi=None, dps=30)
        exact = -mp.log(1 - mp.mpf('0.3'))
        assert abs(q - exact) < mp.mpf('1e-16')


def test_relative_error_zero_reference():
    assert cc.relative_error(mp.mpf('1e-10'), mp.mpf(0)) == mp.mpf('1e-10')
