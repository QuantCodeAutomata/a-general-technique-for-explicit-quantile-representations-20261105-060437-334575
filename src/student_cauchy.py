"""Student's t and Cauchy contour quantile (exp_6).

Implements the general contour method of Schadner (2026) for the Student's t
working density with `a` degrees of freedom (a = 1 is Cauchy; a = 2 has an
elementary closed form used as an independent exact check):

    g(x) = Gamma((a+1)/2) / (sqrt(a*pi) * Gamma(a/2)) * (1 + x^2/a)^(-(a+1)/2)

holomorphically continued to g(z) via the principal branch of the power
(methodology, exp_6), with a CONSTANT contour height

    h(x) = sqrt(a) / (2*(a+1)),   x in Reals

and the BOUNDED observable G(z) = arctan(z) used in the residue form (Eq. 4),
since the identity observable is unbounded and the Student's t tail is only
polynomial (unlike every other law in this repository, whose working density
decays fast enough for the identity observable). The working quantile is
recovered by inverting the bounded observable:

    Q(p) = tan[ -(1/pi) Im sum( arctan(kappa) kappa' g(kappa) / (P(kappa)-p) w ) ]

`student_t_quantile` implements exactly this: call `cc.residue_quantile` with
`observable=lambda z: mp.atan(z)`, then `mp.tan(...)` the result. Location and
scale are reconstructed afterwards, additively/multiplicatively, since this
working law is already the standardized Student's t (methodology, exp_6):
Q~(p) = mu + sigma * Q(p).

Mesh construction (a genuine numerical difficulty specific to this module)
---------------------------------------------------------------------------
Because h(x) is p-independent AND independent of x, the shared engine's
default mesh (`cc.build_mesh` / `cc.build_contour`'s `_one_sided_edges`
floor logic, tuned for heights that shrink monotonically towards a boundary)
is not usable here: its floor term `remaining_distance * 1e-6` is huge
whenever the walk is still far from the (possibly astronomically distant,
see below) right truncation point x_hi, forcing an enormous first cell that
swallows the entire probability bulk near x=0 in a single 20-point
Gauss-Legendre cell. Verified directly: with the shared engine's default
mesh, every evaluated quantile for every (a, p) tested collapses to ~0
(relative error ~1), i.e. it is unusable for this working law, not merely
imprecise.

The residue-form integrand has a bounded but sharply peaked "bump" near
x = x_p (the true quantile): writing P(kappa(x)) = P(x+ih) ~= P(x) + i h g(x)
to first order, 1/(P(kappa(x))-p) is a Poisson-kernel-like peak in x centred
at x_p with HALF-WIDTH ~= h (independent of x_p, since the g(x_p) factors
cancel in the width equation g(x_p) delta_x ~= h g(x_p)). Because h is
constant here, resolving that bump to the same relative accuracy requires
cells no wider than h not just near the origin, but everywhere a quantile of
interest (p in `config.STUDENT_T_PROB_LEVELS`) may fall -- which, for heavy
tails (small a), can be far from the origin (e.g. |Q(0.01)| ~= 1028 for
a=0.5). This was confirmed empirically: subdividing the uniform cell width
further (by 2x, 4x, ...) while covering the same range leaves the computed
quantile unchanged to the digits shown (same stable floor described below),
while *removing* the uniform region (falling back to pure geometric growth)
reproduces the ~1 relative error above.

`build_student_t_contour` therefore builds a custom composite mesh directly
(bypassing `cc.build_mesh`, but still using the shared `cc.leggauss` nodes
and `cc.accumulate`): a UNIFORM region of width exactly h(x) covering
[-x_mid, x_mid], with x_mid set (via a cheap double-precision SciPy estimate
of Q(0.01)/Q(0.99), used only for mesh planning, never as a test oracle) to
comfortably exceed the most extreme quantile magnitude needed, followed by a
GEOMETRIC region (ratio chosen so that exactly `max_cells_per_side` cells
reach the true truncation point x_hi) covering the remaining, rapidly
decaying tail out to x_hi -- which can be enormous for small a (order
exp(60/(a+1)), e.g. ~2.4e17 for a=0.5) since the paper's tail order
O(|x|^-(a+1)) decays only polynomially. This geometric-tail cell count was
verified to not affect the answer: re-running with
`max_cells_per_side` in {50, 150, 400, 800} reproduced the identical quantile
to all digits shown (far tighter than the requested 1e-14 convergence
check), confirming the far tail is already negligible well before
max_cells_per_side=150 -- it is the (separately sized) near-field uniform
region that carries the actual resolution requirement.

Measured accuracy and the tangent-amplification caveat
---------------------------------------------------------------------------
For a in {1, 2, 5, 30} the residue-form quantile matches the high-precision
reference to ~1e-9 or better at every tested p, including p in {0.01, 0.99}.
For a=0.5 at the most extreme levels (p=0.01, 0.99), the measured relative
error floors around 1e-7, and -- confirmed directly -- is *independent* of
further mesh refinement (cell width, far-tail cell count) and of working
precision (24 vs 34 digits): it is the tangent-reconstruction amplification
described in the methodology notes ("amplifies relative errors in the
arctangent observable by a factor of order |quantile|^2"), here
|Q(0.01)|^2 ~= 1028^2 ~= 1.06e6, applied to the underlying (tiny, ~1e-13)
error in the bounded arctan-observable sum itself. Tests use a correspondingly
loosened, measured tolerance for exactly this (a, p) regime -- not a uniform
loosening of every tolerance in this module.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from src import contour_core as cc


def g_student_t(z: mp.mpc, a: float) -> mp.mpc:
    """Holomorphic continuation of the Student's t working density (methodology,
    exp_6), principal branch of the power (valid since Re(1+z^2/a) > 0 on the
    ribbon used: the contour never gets close enough to the branch points
    z = +-i*sqrt(a) for the heights used here)."""
    a_m = mp.mpf(a)
    return (mp.gamma((a_m + 1) / 2) / (mp.sqrt(a_m * mp.pi) * mp.gamma(a_m / 2))
            * mp.power(1 + z ** 2 / a_m, -(a_m + 1) / 2))


def h_student_t(x, a: float) -> mp.mpf:
    """Constant contour height h(x) = sqrt(a) / (2*(a+1)) (methodology, exp_6)."""
    a_m = mp.mpf(a)
    return mp.sqrt(a_m) / (2 * (a_m + 1))


def kappa_student_t(x: mp.mpf, a: float) -> mp.mpc:
    return mp.mpc(mp.mpf(x), h_student_t(x, a))


def kappa_prime_student_t(x: mp.mpf, a: float) -> mp.mpc:
    """kappa'(x) = 1 exactly: h is constant, so Im(kappa) has zero derivative."""
    return mp.mpc(1, 0)


@dataclass
class StudentTContour:
    a: float
    acc: "cc.ContourAccumulation"


def _geometric_tail_edges(x_mid: mp.mpf, x_hi: mp.mpf, n_geo: int):
    """`n_geo` cell edges from x_mid to x_hi (same sign), ratio chosen so the
    last edge lands exactly on x_hi (see module docstring)."""
    ratio = mp.power(x_hi / x_mid, mp.mpf(1) / n_geo)
    edges = [x_mid * mp.power(ratio, k) for k in range(1, n_geo + 1)]
    edges[-1] = x_hi
    return edges


def _build_student_t_mesh(h_const: mp.mpf, n_uniform: int, x_hi: mp.mpf,
                           n_geo: int, gl_n: int = cc.GL_N) -> cc.Mesh:
    """Symmetric mesh: uniform cells of width `h_const` on [-x_mid, x_mid]
    (x_mid = n_uniform * h_const), then `n_geo` geometric cells on each side
    out to +-x_hi (see module docstring for the rationale)."""
    nodes_unit, weights_unit = cc.leggauss(gl_n)
    uniform_edges = [h_const * k for k in range(n_uniform + 1)]
    x_mid = uniform_edges[-1]
    right_edges = uniform_edges + _geometric_tail_edges(x_mid, x_hi, n_geo)
    left_edges = [-e for e in reversed(right_edges)]
    edges = left_edges[:-1] + right_edges
    xs, ws = [], []
    for a_edge, b_edge in zip(edges[:-1], edges[1:]):
        mid, half = (a_edge + b_edge) / 2, (b_edge - a_edge) / 2
        for t, wt in zip(nodes_unit, weights_unit):
            xs.append(mid + half * t)
            ws.append(half * wt)
    return cc.Mesh(xs=xs, ws=ws, x_lo=-x_hi, x_hi=x_hi, edges=edges, gl_n=gl_n)


def build_student_t_contour(a: float, dps: int = 24, max_cells_per_side: int = 200,
                             safety: float = 1.25) -> StudentTContour:
    """Build the p-independent Student's t(a) contour once (methodology, exp_6).

    `max_cells_per_side` bounds only the GEOMETRIC tail cell count (reaching
    the true truncation x_hi, which can be astronomically far for small a);
    verified not to affect the answer at all for max_cells_per_side >= 50
    (see module docstring). The near-field UNIFORM cell count is sized
    separately and automatically from a cheap double-precision SciPy estimate
    of Q(0.01)/Q(0.99) (mesh planning only, never used as a reference/oracle
    anywhere in this module), inflated by `safety`, to guarantee every
    quantile needed for `config.STUDENT_T_PROB_LEVELS` lands inside the
    uniformly-resolved region (see module docstring for why this is required).
    """
    with mp.workdps(dps):
        from scipy.stats import t as scipy_t

        a_m = mp.mpf(a)
        h_const = h_student_t(mp.mpf(0), a)
        q_extreme = max(abs(scipy_t.ppf(0.01, a)), abs(scipy_t.ppf(0.99, a)))
        n_uniform = int(mp.ceil(mp.mpf(safety) * mp.mpf(q_extreme) / h_const))

        def order_fn(x):
            # density tail order O(|x|^-(a+1)) (methodology, exp_6)
            return abs(x) ** -(a_m + 1)

        x_hi = cc.find_right_truncation(order_fn, mp.mpf(10), target_log=60.0)
        mesh = _build_student_t_mesh(h_const, n_uniform, x_hi, max_cells_per_side)

        def kappa(x):
            return kappa_student_t(x, a)

        def kappa_prime(x):
            return kappa_prime_student_t(x, a)

        def g(z):
            return g_student_t(z, a)

        acc = cc.accumulate(mesh, kappa, kappa_prime, g, p_left=mp.mpf(0))
        return StudentTContour(a=a, acc=acc)


def student_t_quantile(contour: StudentTContour, p: float, dps: int = 24) -> mp.mpf:
    """Student's t(a) working quantile Q(p) via the residue form (Eq. 4) with
    the bounded arctan observable (methodology, exp_6): calls
    `cc.residue_quantile(contour.acc, p, observable=lambda z: mp.atan(z))`,
    which returns arctan(Q(p)), then applies `mp.tan(...)` to recover Q(p)
    itself. This is the raw working quantile; combine with `mu`/`sigma` via
    `student_t_quantile_scaled` for the located/scaled quantile."""
    with mp.workdps(dps):
        raw = cc.residue_quantile(contour.acc, mp.mpf(p), observable=lambda z: mp.atan(z))
        return mp.tan(raw)


def student_t_quantile_scaled(contour: StudentTContour, p: float, mu: float = 0.0,
                               sigma: float = 1.0, dps: int = 24) -> mp.mpf:
    """Q~(p) = mu + sigma * Q(p) (methodology, exp_6)."""
    with mp.workdps(dps):
        q = student_t_quantile(contour, p, dps=dps)
        return mp.mpf(mu) + mp.mpf(sigma) * q


def cauchy_exact_quantile(p: float) -> mp.mpf:
    """Exact closed form for a=1 (Cauchy), kappa_1(x) = x + i/4 (methodology,
    exp_6): Q(p) = tan(pi*(p - 1/2)), no contour/quadrature involved."""
    return mp.tan(mp.pi * (mp.mpf(p) - mp.mpf('0.5')))


def student_t2_exact_quantile(p: float) -> mp.mpf:
    """Exact closed form for a=2, kappa_2(x) = x + i*sqrt(2)/6 (methodology,
    exp_6): Q(p) = (2p-1) / sqrt(2p(1-p)), no contour/quadrature involved."""
    p_m = mp.mpf(p)
    return (2 * p_m - 1) / mp.sqrt(2 * p_m * (1 - p_m))


def cdf_student_t(x: mp.mpf, a: float) -> mp.mpf:
    """Real CDF via the regularized incomplete beta function (methodology,
    exp_6): P(x) = 1 - 0.5*I_{a/(a+x^2)}(a/2, 1/2) for x>0, and 1-P(-x) for
    x<=0 (equivalently 0.5*I_{a/(a+x^2)}(a/2, 1/2), since a/(a+x^2) is even
    in x)."""
    a_m, x_m = mp.mpf(a), mp.mpf(x)
    xt = a_m / (a_m + x_m ** 2)
    ib = mp.betainc(a_m / 2, mp.mpf('0.5'), 0, xt, regularized=True)
    return 1 - ib / 2 if x_m > 0 else ib / 2


def reference_quantile_student_t(a: float, p: float, dps: int = 30) -> mp.mpf:
    """Reference quantile via SciPy-seeded Newton's method in high precision
    (Appendix D.1), refining against the betainc-based real CDF `cdf_student_t`."""
    from scipy.stats import t as scipy_t
    with mp.workdps(dps):
        x0 = float(scipy_t.ppf(p, a))
        return cc.newton_quantile(lambda x: cdf_student_t(x, a), p, x0, lo=None, hi=None, dps=dps)
