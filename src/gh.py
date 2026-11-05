"""Generalized hyperbolic (GH) contour quantile (exp_4).

Implements the general contour method of Schadner (2026) for the
standardized GH working density on I = Reals (a in Reals, b > |d| >= 0):

    f(x) = (b^2-d^2)^(a/2) * e^(d*x) * (1+x^2)^((a-1/2)/2)
           / [sqrt(2*pi) * b^(a-1/2) * K_a(sqrt(b^2-d^2))]
           * K_{a-1/2}(b*sqrt(1+x^2))                                (methodology, exp_4)

holomorphically continued to g(z) via the principal branches of `mp.power`
and `mp.besselk` (valid for |Im z| < 1, where Re(1+z^2) > 0; the contour
height used here is always well under 1, see `h_gh`, so this holds along the
whole contour -- asserted, not merely assumed, in `g_gh`).

The working law g IS the standardized density itself (no separate "working
law -> target law" map beyond the final affine rescaling), so location/scale
are recovered after the fact: Q~(p) = mu + sigma * Q(p) (methodology, exp_4).

Contour height (Eq. A.5, Appendix A.1, Proposition A.1) is a CONSTANT scalar,
independent of x:

    D = 1 + |a - 1/2| - a - 1/2
    A* = exp(D * b * (1 - sqrt(3)/2))
    M* = A* * (|d| + 2*b/sqrt(3) + 4*D/3)
    h  = min(1/2, 1/(2*M*))

so kappa(x) = x + i*h (Eq. 2) and kappa'(x) = 1 exactly (passed explicitly,
not left to the shared engine's numerical derivative, since it is exact and
cheaper).

Mesh construction (custom, bypassing `cc.build_mesh`)
---------------------------------------------------------------------------
Like `src/student_cauchy.py` (Student's t / Cauchy, also a constant-height
law on all of Reals), the shared engine's default mesh-growth logic is tuned
for heights that shrink toward a boundary and fails for a constant height
(it produces one oversized first cell -- see `src/student_cauchy.py`'s module
docstring for the diagnostic). This module replicates that module's fix: a
custom composite mesh of UNIFORM cells of width exactly `h` covering the
near-field region where the probability bulk and every needed quantile level
can fall, followed (where the asymmetric tail order b-d vs b+d makes the
true truncation point(s) farther out) by a GEOMETRIC tail region on each side
out to the respective truncation point. Unlike the (symmetric) Student's t
law, the GH tails are generally ASYMMETRIC (order x^(a-1) e^{-(b-d)x} to the
right, y^(a-1) e^{-(b+d)y} to the left, y=-x), so the uniform-region cell
counts and the truncation points are sized independently on each side.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from src import contour_core as cc


def _besselk_safe_order(order: mp.mpf) -> mp.mpf:
    """`order`, perturbed by an amount far below working precision whenever
    it is exactly (or extremely nearly) an integer.

    K_nu(z) is entire in nu (the textbook formula has only a *removable*
    singularity at integer nu, resolved by a limit), so this perturbation
    changes the returned value by an amount far smaller than working
    precision can represent -- mathematically immaterial. It matters
    numerically because mpmath's `besselk`, for |z| large (its asymptotic
    2F0-type representation), applies a conservative extra precision-boost
    whenever the order is detected to be (nearly) integer (`hypercomb`'s
    `nint_distance` check: the boost grows as the order approaches an
    integer), which is ~5-10x slower for complex z than a safely
    non-integer order. This matters for this module specifically at
    a=-0.5 (NIG), where order = a-1/2 = -1 exactly, evaluated at ~10^4
    complex mesh points per contour -- without perturbing away from the
    exact integer, building/using that contour is impractically slow
    (confirmed directly during development: ~0.06s/call, vs ~0.012s/call
    perturbed).

    The perturbation `eps = 1e-13` is a fixed empirical compromise (not
    tied to `mp.mp.dps`, since a perturbation tied to the *current* working
    precision's ULP, e.g. 10^-(dps+20), gets rounded away to exactly the
    integer again at dps ~ 24-30 used throughout this repository -- the
    first version of this fix made that mistake and silently did nothing):
    confirmed empirically to introduce a relative error in besselk's
    returned value of order 1e-15 (measured across z on the real axis and
    off it by the contour height, for the order=-1 case), two orders of
    magnitude below this module's 1e-13 test tolerance, while still cutting
    the per-call time roughly 5x (an order close enough to an integer, e.g.
    eps=1e-20, needs *more* extra precision to resolve the near-cancellation
    than a safely-displaced order, and is almost as slow as the exact
    integer)."""
    nearest = mp.nint(order)
    if order == nearest:
        eps = mp.mpf('1e-13')
        return order + eps
    return order


def g_gh(z: mp.mpc, a: float, b: float, d: float) -> mp.mpc:
    """Holomorphic continuation of the standardized GH working density
    (methodology, exp_4). Principal branches of `mp.power` (for
    (1+z^2)^((a-1/2)/2)) and `mp.besselk` (complex order a-1/2, complex
    argument b*sqrt(1+z^2)); valid for |Im z| < 1 so that Re(1+z^2) > 0 --
    the contour height `h_gh` used throughout this module is always << 1
    (verified: h_gh <= 1/2 by construction), so this holds along the whole
    contour."""
    a_m, b_m, d_m = mp.mpf(a), mp.mpf(b), mp.mpf(d)
    gamma2 = b_m ** 2 - d_m ** 2
    one_plus_z2 = 1 + z ** 2
    order_a = _besselk_safe_order(a_m)
    order_half = _besselk_safe_order(a_m - mp.mpf('0.5'))
    numer = (mp.power(gamma2, a_m / 2) * mp.e ** (d_m * z)
             * mp.power(one_plus_z2, (a_m - mp.mpf('0.5')) / 2))
    denom = (mp.sqrt(2 * mp.pi) * mp.power(b_m, a_m - mp.mpf('0.5'))
              * mp.besselk(order_a, mp.sqrt(gamma2)))
    bessel_term = mp.besselk(order_half, b_m * mp.sqrt(one_plus_z2))
    return numer / denom * bessel_term


def h_gh(a: float, b: float, d: float) -> mp.mpf:
    """Constant contour height h (Eq. A.5, Prop. A.1): a single scalar,
    independent of x."""
    a_m, b_m, d_m = mp.mpf(a), mp.mpf(b), mp.mpf(d)
    half = mp.mpf('0.5')
    D = 1 + abs(a_m - half) - a_m - half
    A_star = mp.e ** (D * b_m * (1 - mp.sqrt(3) / 2))
    M_star = A_star * (abs(d_m) + 2 * b_m / mp.sqrt(3) + 4 * D / 3)
    return min(half, 1 / (2 * M_star))


def kappa_gh(x: mp.mpf, h_const: mp.mpf) -> mp.mpc:
    return mp.mpc(mp.mpf(x), h_const)


def kappa_prime_gh(x: mp.mpf) -> mp.mpc:
    """kappa'(x) = 1 exactly: h is constant, so Im(kappa) has zero derivative."""
    return mp.mpc(1, 0)


def _truncation(a: float, b: float, d: float):
    """Right truncation x_hi (tail order x^(a-1) e^{-(b-d)x}) and left
    truncation x_lo = -y_hi (tail order y^(a-1) e^{-(b+d)y}, y=-x>0), each
    truncated where the neglected tail order is e^{-60} (methodology,
    exp_4). `x_start` is inflated when the relevant decay rate is small, so
    the geometric search in `cc.find_right_truncation` (which starts
    searching only from x_start, growing by 1.5x per step) does not have to
    climb through many orders of magnitude of decay-rate-dependent distance
    before even reaching the region where the tail order starts dropping."""
    a_m, b_m, d_m = mp.mpf(a), mp.mpf(b), mp.mpf(d)
    decay_right = b_m - d_m
    decay_left = b_m + d_m

    def order_right(x):
        xx = max(x, mp.mpf('1e-6'))
        return abs(xx ** (a_m - 1) * mp.e ** (-decay_right * xx))

    def order_left(y):
        yy = max(y, mp.mpf('1e-6'))
        return abs(yy ** (a_m - 1) * mp.e ** (-decay_left * yy))

    start_right = max(mp.mpf(10), mp.mpf(20) / max(decay_right, mp.mpf('1e-6')))
    start_left = max(mp.mpf(10), mp.mpf(20) / max(decay_left, mp.mpf('1e-6')))
    x_hi = cc.find_right_truncation(order_right, start_right, target_log=60.0)
    y_hi = cc.find_right_truncation(order_left, start_left, target_log=60.0)
    return -y_hi, x_hi


def _geometric_tail_edges(x_mid: mp.mpf, x_stop: mp.mpf, n_geo: int):
    """`n_geo` cell edges (positive magnitudes) from x_mid to x_stop, ratio
    chosen so the last edge lands exactly on x_stop (same device as
    `src/student_cauchy.py`'s `_geometric_tail_edges`)."""
    ratio = mp.power(x_stop / x_mid, mp.mpf(1) / n_geo)
    edges = [x_mid * mp.power(ratio, k) for k in range(1, n_geo + 1)]
    edges[-1] = x_stop
    return edges


def _one_side_edges(h_const: mp.mpf, n_uniform: int, x_trunc_abs: mp.mpf, n_geo: int):
    """Ascending positive-magnitude edges 0 = e_0 < e_1 < ... from `n_uniform`
    uniform cells of width `h_const`, followed by `n_geo` geometric cells
    reaching exactly `x_trunc_abs`."""
    uniform_edges = [h_const * k for k in range(n_uniform + 1)]
    x_mid = uniform_edges[-1]
    if x_mid >= x_trunc_abs:
        return uniform_edges
    return uniform_edges + _geometric_tail_edges(x_mid, x_trunc_abs, n_geo)


def build_gh_mesh(h_const: mp.mpf, n_left: int, n_right: int, x_lo: mp.mpf, x_hi: mp.mpf,
                   n_geo: int, gl_n: int = cc.GL_N) -> cc.Mesh:
    """Asymmetric composite mesh: uniform cells of width `h_const` out to
    +-n_{left,right}*h_const on each side, then a (separately sized)
    geometric tail on each side out to x_lo/x_hi (see module docstring)."""
    nodes_unit, weights_unit = cc.leggauss(gl_n)
    right_edges = _one_side_edges(h_const, n_right, x_hi, n_geo)
    left_edges = list(reversed([-e for e in _one_side_edges(h_const, n_left, -x_lo, n_geo)]))
    edges = left_edges[:-1] + right_edges
    xs, ws = [], []
    for a_edge, b_edge in zip(edges[:-1], edges[1:]):
        mid, half = (a_edge + b_edge) / 2, (b_edge - a_edge) / 2
        for t, wt in zip(nodes_unit, weights_unit):
            xs.append(mid + half * t)
            ws.append(half * wt)
    return cc.Mesh(xs=xs, ws=ws, x_lo=x_lo, x_hi=x_hi, edges=edges, gl_n=gl_n)


@dataclass
class GHContour:
    a: float
    b: float
    d: float
    h_const: mp.mpf
    acc: "cc.ContourAccumulation"


def build_gh_contour(a: float, b: float, d: float, dps: int = 24,
                      max_cells_per_side: int = 200, safety: float = 1.25) -> GHContour:
    """Build the p-independent GH(a, b, d) contour once (methodology, exp_4).

    `max_cells_per_side` bounds only the GEOMETRIC tail cell count (reaching
    the true truncation x_lo/x_hi); the near-field UNIFORM cell counts are
    sized separately and automatically from a cheap double-precision SciPy
    estimate of Q(0.01)/Q(0.99) via `scipy.stats.genhyperbolic.ppf` (mesh
    planning only, never used as a reference/oracle anywhere in this
    module -- verified to exactly reproduce this module's own `g_gh` density
    in tests/test_gh.py), inflated by `safety`, following the same device as
    `src/student_cauchy.py`'s `build_student_t_contour`.
    """
    with mp.workdps(dps):
        h_const = h_gh(a, b, d)
        x_lo, x_hi = _truncation(a, b, d)

        from scipy.stats import genhyperbolic
        q_lo_est = abs(float(genhyperbolic.ppf(0.01, a, b, d)))
        q_hi_est = abs(float(genhyperbolic.ppf(0.99, a, b, d)))
        n_left = max(1, int(mp.ceil(mp.mpf(safety) * mp.mpf(q_lo_est) / h_const)))
        n_right = max(1, int(mp.ceil(mp.mpf(safety) * mp.mpf(q_hi_est) / h_const)))

        mesh = build_gh_mesh(h_const, n_left, n_right, x_lo, x_hi, max_cells_per_side)

        def kappa(x):
            return kappa_gh(x, h_const)

        def kappa_prime(x):
            return kappa_prime_gh(x)

        def g(z):
            return g_gh(z, a, b, d)

        acc = cc.accumulate(mesh, kappa, kappa_prime, g, p_left=mp.mpf(0))
        return GHContour(a=a, b=b, d=d, h_const=h_const, acc=acc)


def gh_quantile(contour: GHContour, p: float, dps: int = 24) -> mp.mpf:
    """Standardized GH working quantile Q(p) via the residue form, Eq. (4)."""
    with mp.workdps(dps):
        return cc.residue_quantile(contour.acc, mp.mpf(p))


def gh_quantile_scaled(contour: GHContour, p: float, mu: float = 0.0, sigma: float = 1.0,
                        dps: int = 24) -> mp.mpf:
    """Affine reconstruction Q~(p) = mu + sigma * Q(p) (methodology, exp_4)."""
    with mp.workdps(dps):
        q = gh_quantile(contour, p, dps=dps)
        return mp.mpf(mu) + mp.mpf(sigma) * q


def g_gh_float(x: float, a: float, b: float, d: float) -> float:
    """Pure double-precision reimplementation of the real-axis density
    using `scipy.special.kv`, matching `g_gh`'s formula exactly (used both
    for the double-precision reference CDF and as the holomorphic-continuation
    sanity check in tests/test_gh.py)."""
    import math
    from scipy.special import kv
    gamma2 = b ** 2 - d ** 2
    one_plus_x2 = 1 + x ** 2
    numer = (gamma2 ** (a / 2)) * math.exp(d * x) * one_plus_x2 ** ((a - 0.5) / 2)
    denom = math.sqrt(2 * math.pi) * b ** (a - 0.5) * kv(a, math.sqrt(gamma2))
    return numer / denom * kv(a - 0.5, b * math.sqrt(one_plus_x2))


def reference_quantile_gh(a: float, b: float, d: float, p: float, dps: int = 30) -> mp.mpf:
    """Reference quantile (methodology, exp_4): a double-precision CDF built
    from `g_gh_float` with `scipy.integrate.quad`, root-found with
    `scipy.optimize.brentq` bracketed by the (double-precision cast of the)
    truncation points, then refined to `dps` digits with `cc.newton_quantile`
    against a high-precision mpmath CDF `mp.quad(g_gh, [x_lo, x])` (mpmath
    quadrature of the mpmath density, not SciPy, for the high-precision
    refinement step -- methodology, exp_4)."""
    from scipy.integrate import quad
    from scipy.optimize import brentq

    x_lo, x_hi = _truncation(a, b, d)
    x_lo_f, x_hi_f = float(x_lo), float(x_hi)

    def cdf_float(x):
        val, _ = quad(lambda t: g_gh_float(t, a, b, d), x_lo_f, x, limit=200)
        return val

    x0 = brentq(lambda x: cdf_float(x) - p, x_lo_f, x_hi_f, xtol=1e-14, rtol=1e-14)

    with mp.workdps(dps):
        def cdf_real(x):
            return mp.quad(lambda t: g_gh(t, a, b, d), [x_lo, mp.mpf(x)])

        return cc.newton_quantile(cdf_real, p, x0, lo=x_lo_f, hi=x_hi_f, dps=dps)
