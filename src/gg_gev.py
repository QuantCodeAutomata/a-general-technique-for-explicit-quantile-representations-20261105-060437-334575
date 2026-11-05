"""Generalized Gamma (GG) and Generalized Extreme Value (GEV) quantile
construction (exp_3).

Implements the general contour method of Schadner (2026) for a single
working law -- the unit-rate gamma(a) distribution on I=(0,infinity)

    g(x) = x^(a-1) * e^(-x) / Gamma(a),   x > 0                 (methodology, exp_3)

with contour height shared by both reconstructions (GG with shape a=a, GEV
with shape a=1, i.e. the exponential law)

    h(x) = x / (1 + |a-1| + x),   x > 0                         (methodology, exp_3)

kappa(x) = x + i*h(x); kappa'(x) is left to `cc.build_contour`'s default
numerical (central-difference) derivative, as instructed by the methodology
-- unlike the GIG/Beta modules, there is no domain boundary near x=0 for `h`
itself (it has no singularity there), so probing slightly outside (0,
infinity) during differencing is harmless.

GG (Generalized Gamma): the increasing map u = sigma * x^(1/c) takes the
gamma(a) working law to the standardized GG density
f(x) = c/Gamma(a) * x^(ca-1) * e^(-x^c). Reconstruction:

    Q~_GG(p) = sigma * [Q(p)]^(1/c)                              (methodology, exp_3)

GEV (Generalized Extreme Value): the DECREASING map
u = mu + sigma*(x^(-a) - 1)/a (a != 0), u = mu - sigma*log(x) (a=0), takes
the unit-rate EXPONENTIAL law (gamma(a=1)) to GEV. Because the map is
decreasing, the reconstruction evaluates the gamma(a=1) working quantile at
(1-p), not p:

    Q~_GEV(p) = map(Q_exp(1-p); a, sigma, mu)                    (methodology, exp_3)

which reduces to the elementary closed form (since Q_exp(1-p) = -log(p)
exactly, the exponential CDF being 1 - e^{-x}):

    Q~_GEV(p) = mu + (sigma/a) * ((-log p)^(-a) - 1),   a != 0
    Q~_GEV(p) = mu - sigma * log(-log p),                a = 0

This elementary closed form is the paper's own reduction and is treated as
ground truth for GEV throughout this module (see `reference_quantile_gev`);
it is NOT the same as SciPy's `genextreme`, whose shape parameter `c` is
opposite-signed to this paper's `a` (c = -a) and is used only as an
independent sanity check in the experiment script, never as a test oracle.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from src import contour_core as cc

GG_SIGMA = 1.0  # implicit scale for config.GG_TEST_PARAMS (a, c) pairs


def g_gamma(z: mp.mpc, a: float) -> mp.mpc:
    """Holomorphic continuation of the unit-rate gamma(a) working density
    (methodology, exp_3), using the principal branch of z^(a-1) continued
    from the positive real axis."""
    a_m = mp.mpf(a)
    return mp.power(z, a_m - 1) * mp.e ** (-z) / mp.gamma(a_m)


def h_gamma(x: mp.mpf, a: float) -> mp.mpf:
    """Contour height shared by GG (a=a) and GEV (a=1) (methodology, exp_3)."""
    a_m = mp.mpf(a)
    x = mp.mpf(x)
    return x / (1 + abs(a_m - 1) + x)


@dataclass
class GammaContour:
    a: float
    acc: "cc.ContourAccumulation"


def _mesh_kwargs_for(a: float, max_cells_per_side: int) -> dict:
    """Mesh tuning depending on |a-1| (see `build_gamma_contour` docstring).

    Close to a=1 (|a-1| < 0.5, which covers every GEV case -- always built
    from the a=1 contour -- and GG's own a=1 test cases), h(x) = x/(1+x)
    makes the left mesh shrink quadratically near zero, leaving a single
    composite-GL cell spanning close to a full decade of x near the smallest
    resolved quantiles at the shared engine's default width_factor=1.0; a
    narrower width_factor=0.2 fixes this without needing more cells than the
    default elsewhere (x_lo is not extreme there: order x^a with a near 1
    gives x_lo of order e^-60, not e^-60/a for small a). Away from a=1, the
    default width_factor=1.0 both resolves the (potentially very small, e.g.
    ~e^-150 for a=0.4) x_lo within max_cells_per_side and avoids needing
    more cells than that budget allows (narrowing width_factor there instead
    would need several times max_cells_per_side just to reach x_lo).
    """
    a_m = mp.mpf(a)
    if abs(a_m - 1) < mp.mpf('0.5'):
        return dict(width_factor=mp.mpf('0.2'), growth=mp.mpf('1.05'),
                    max_cells_per_side=max_cells_per_side)
    return dict(width_factor=mp.mpf('1.0'), growth=mp.mpf('1.1'),
                max_cells_per_side=max_cells_per_side)


def build_gamma_contour(a: float, dps: int = 24, max_cells_per_side: int = 160,
                         guard_digits: int = 10) -> GammaContour:
    """Build the p-independent gamma(a) working contour once (methodology, exp_3).

    Truncation uses the working-quantile kernel order, not the bare density
    order: the residue-form integrand kappa*kappa'*g(kappa) is O(x^a) near
    zero (kappa ~ x, kappa' ~ 1, g(kappa) ~ x^(a-1)/Gamma(a)) and
    O(x^a e^{-x}) at infinity (methodology, exp_3).

    `guard_digits=10` extra accumulation precision (see `gamma_quantile`) is
    needed everywhere to beat the shared engine's *fixed*-step default
    numerical derivative (`cc.numerical_derivative`, step ~ 10^-(dps//2)),
    whose error is otherwise the accuracy floor once the mesh itself is fine
    enough (confirmed directly: refining the mesh without raising the
    working precision plateaus around 1e-14, independent of mesh size).
    `_mesh_kwargs_for` additionally narrows the mesh near a=1 (see its
    docstring) for the same reason.

    Known, measured limitation (same character as the Beta a<1/b<1 case in
    `src/gb.py`): for a < 1, g(kappa(x)) ~ x^(a-1) diverges (integrably) at
    the left endpoint, and the composite Gauss-Legendre quadrature cell
    nearest that singularity converges only slowly with mesh/precision
    refinement -- e.g. a=0.4 plateaus around 2e-14..5e-14 relative error
    regardless of guard_digits, confirming genuine quadrature error rather
    than rounding (see tests/test_gg_gev.py, which documents the measured
    tolerance for this case rather than asserting the generic <1e-15 bound).
    """
    with mp.workdps(dps + guard_digits):
        a_m = mp.mpf(a)

        def h(x):
            return h_gamma(x, a)

        def g(z):
            return g_gamma(z, a)

        def left_order(x):
            return abs(mp.power(x, a_m))

        def right_order(x):
            return abs(mp.power(x, a_m) * mp.e ** (-x))

        x_start_right = max(mp.mpf(1), a_m)
        x_hi = cc.find_right_truncation(right_order, x_start_right, target_log=60.0)
        x_lo = cc.find_left_truncation(left_order, mp.mpf('1.0'), target_log=60.0,
                                        x_min=mp.mpf('1e-300'))
        center = max(mp.mpf(1), min(x_hi / 4, mp.mpf(10)))
        acc = cc.build_contour(g, h, center, x_lo, x_hi,
                                mesh_kwargs=_mesh_kwargs_for(a, max_cells_per_side))
        return GammaContour(a=a, acc=acc)


def gamma_quantile(contour: GammaContour, p: float, dps: int = 24, guard_digits: int = 10) -> mp.mpf:
    """Gamma(a) working quantile Q(p) via the residue form, Eq. (4).

    Evaluated at `dps + guard_digits` working precision, matching the
    accumulation precision used in `build_gamma_contour` (see its docstring).
    """
    with mp.workdps(dps + guard_digits):
        return cc.residue_quantile(contour.acc, mp.mpf(p))


# --- Generalized Gamma (GG) reconstruction ---

def gg_quantile(contour: GammaContour, p: float, c: float, sigma: float = GG_SIGMA,
                 dps: int = 24, guard_digits: int = 10) -> mp.mpf:
    """GG quantile: Q~_GG(p) = sigma * [Q(p)]^(1/c) (methodology, exp_3).

    Evaluated at `dps + guard_digits` working precision, matching the
    accumulation precision used in `build_gamma_contour` (see its docstring).
    """
    with mp.workdps(dps + guard_digits):
        q = cc.residue_quantile(contour.acc, mp.mpf(p))
        return mp.mpf(sigma) * mp.power(q, 1 / mp.mpf(c))


def reference_quantile_gg(a: float, c: float, p: float, sigma: float = GG_SIGMA,
                           dps: int = 30) -> mp.mpf:
    """GG reference quantile via SciPy-seeded Newton's method in high precision.

    SciPy's `gengamma(a, c)` pdf is `c * x**(c*a-1) * exp(-x**c) / gamma(a)`
    for x > 0, matching the standardized GG density in this module exactly
    (methodology, exp_3); the real CDF used to refine the SciPy seed is
    obtained by `mp.quad` integrating that (sigma-scaled) density from a tiny
    left cutoff up to x.
    """
    from scipy.stats import gengamma
    with mp.workdps(dps):
        a_m, c_m, sigma_m = mp.mpf(a), mp.mpf(c), mp.mpf(sigma)
        x0 = float(gengamma.ppf(p, a, c)) * float(sigma)

        def pdf_real(t):
            u = t / sigma_m
            return (c_m / mp.gamma(a_m)) * mp.power(u, c_m * a_m - 1) * mp.e ** (-mp.power(u, c_m)) / sigma_m

        def cdf_real(x):
            return mp.quad(pdf_real, [mp.mpf('1e-300'), mp.mpf(x)])

        return cc.newton_quantile(cdf_real, p, x0, lo=1e-300, hi=None, dps=dps)


# --- Generalized Extreme Value (GEV) reconstruction ---

def gev_map(x: mp.mpf, a: float, sigma: float, mu: float) -> mp.mpf:
    """Decreasing map taking the unit-rate exponential variable x to GEV
    (methodology, exp_3):
        u = mu + sigma*(x^(-a) - 1)/a,   a != 0
        u = mu - sigma*log(x),            a = 0
    """
    a_m, sigma_m, mu_m = mp.mpf(a), mp.mpf(sigma), mp.mpf(mu)
    x_m = mp.mpf(x)
    if a_m == 0:
        return mu_m - sigma_m * mp.log(x_m)
    return mu_m + (sigma_m / a_m) * (mp.power(x_m, -a_m) - 1)


def gev_quantile(contour: GammaContour, p: float, a: float, sigma: float, mu: float,
                  dps: int = 24, guard_digits: int = 10) -> mp.mpf:
    """GEV quantile via the contour chain (methodology, exp_3): evaluates the
    gamma(a=1) [exponential] working quantile at level (1-p) because the GEV
    map is decreasing, then applies `gev_map`. `contour` must be a
    `GammaContour` built with a=1.0 (`build_gamma_contour(1.0, ...)`).

    Evaluated at `dps + guard_digits` working precision, matching the
    accumulation precision used in `build_gamma_contour` (see its docstring;
    every GEV case uses the a=1 contour, exactly the regime that needs the
    extra guard digits)."""
    assert contour.a == 1.0, "gev_quantile requires the gamma(a=1) [exponential] contour"
    with mp.workdps(dps + guard_digits):
        q1 = cc.residue_quantile(contour.acc, 1 - mp.mpf(p))
        return gev_map(q1, a, sigma, mu)


def reference_quantile_gev(p: float, a: float, sigma: float, mu: float, dps: int = 30) -> mp.mpf:
    """GEV reference quantile: the elementary closed form the paper's own
    reconstruction reduces to (methodology, exp_3), trusted as ground truth
    throughout this module -- NOT SciPy's `genextreme` (opposite-signed shape
    convention, c = -a; used only for an independent sanity check elsewhere).
    Equivalent to `gev_map(-log(p), a, sigma, mu)` since Q_exp(1-p) = -log(p)
    exactly (the exponential CDF is 1 - e^{-x}).
    """
    with mp.workdps(dps):
        x = -mp.log(mp.mpf(p))
        return gev_map(x, a, sigma, mu)


def genextreme_sanity_check(p: float, a: float, sigma: float, mu: float) -> float:
    """Independent double-precision sanity check via SciPy's `genextreme`,
    whose shape parameter `c` is opposite-signed to this paper's `a`
    (methodology, exp_3). Informational only -- never used as a test oracle."""
    from scipy.stats import genextreme
    return float(genextreme.ppf(p, -a, loc=mu, scale=sigma))
