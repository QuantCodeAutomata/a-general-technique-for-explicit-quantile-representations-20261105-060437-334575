"""Normal and log-normal contour quantile construction (exp_5).

Implements the general contour method of Schadner (2026) for the standard
normal working density on I = Reals

    g(x) = phi(x) = e^(-x^2/2) / sqrt(2*pi),   x in Reals    (methodology, exp_5)

with contour height (Eq. A.10)

    h(x) = 1 / (2*(1+x^2)),   x in Reals                     (methodology, exp_5)

kappa(x) = x + i*h(x); kappa'(x) is left to `cc.build_contour`'s default
numerical (central-difference) derivative -- g and h are both entire/smooth
on the whole real line, so there is no domain boundary to worry about when
probing slightly off the real axis during differencing.

Normal: the affine map u = mu + sigma*x takes the standard normal working
law to Normal(mu, sigma^2). Reconstruction:

    Q~(p) = mu + sigma * Q(p)                                 (methodology, exp_5)

Log-normal: the increasing map u = sigma * exp(a*x) (sigma = median, a = the
log-scale standard deviation) takes the SAME standard normal working law to
the standardized log-normal density f(x) = 1/(a*x*sqrt(2*pi)) *
exp(-(log x)^2/(2*a^2)), x > 0. Reconstruction:

    Q~(p) = sigma * exp(a * Q(p))                             (methodology, exp_5)

reusing the identical p-independent standard-normal contour built once for
the plain normal case above -- the whole point of this experiment is that
log-normal quantiles come from the same contour via a different
reconstruction map, not a separately built one.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from src import contour_core as cc


def g_normal(z: mp.mpc) -> mp.mpc:
    """Holomorphic continuation of the standard normal working density
    (methodology, exp_5): g(z) = e^(-z^2/2)/sqrt(2*pi), entire, no branch issues."""
    return mp.e ** (-z ** 2 / 2) / mp.sqrt(2 * mp.pi)


def h_normal(x: mp.mpf) -> mp.mpf:
    """Contour height (Eq. A.10): h(x) = 1/(2*(1+x^2)), bounded on Reals."""
    x = mp.mpf(x)
    return 1 / (2 * (1 + x ** 2))


@dataclass
class NormalContour:
    """The p-independent standard-normal contour, built once and reused for
    both the plain normal (affine map) and log-normal (exponential map)
    reconstructions."""

    acc: "cc.ContourAccumulation"


def build_normal_contour(dps: int = 24, max_cells_per_side: int = 160,
                          guard_digits: int = 12) -> NormalContour:
    """Build the p-independent standard-normal working contour once
    (methodology, exp_5).

    Truncation uses the Gaussian tail order function order_fn(x) =
    abs(exp(-x^2/2)) starting from x_start=5 (methodology, exp_5). Since the
    domain is the whole real line and the tail order function is symmetric
    (even in x), the mesh is seeded at center=0 with x_lo = -x_hi by
    symmetry, rather than calling `cc.find_left_truncation` (designed for
    x > 0 "smallest-magnitude x" searches, not a two-sided domain).

    `guard_digits=12` extra accumulation precision (see `normal_quantile`) is
    needed to beat the shared engine's *fixed*-step default numerical
    derivative (`cc.numerical_derivative`, step ~ 10^-(dps//2)), which is
    otherwise the accuracy floor well above the 1e-18 target once the mesh
    itself is fine enough (confirmed directly: refining the mesh alone
    without raising the working precision plateaus around 2.6e-14 relative
    error, same character as the guard-digit requirement documented in
    `src/gg_gev.py`, `build_gamma_contour`).
    """
    with mp.workdps(dps + guard_digits):
        def h(x):
            return h_normal(x)

        def g(z):
            return g_normal(z)

        def order_fn(x):
            return abs(mp.e ** (-x ** 2 / 2))

        x_hi = cc.find_right_truncation(order_fn, mp.mpf(5), target_log=60.0)
        x_lo = -x_hi
        acc = cc.build_contour(g, h, mp.mpf(0), x_lo, x_hi,
                                p_left=mp.mpf(0),
                                mesh_kwargs=dict(max_cells_per_side=max_cells_per_side))
        return NormalContour(acc=acc)


def normal_standard_quantile(contour: NormalContour, p: float, dps: int = 24,
                              guard_digits: int = 12) -> mp.mpf:
    """Standard-normal working quantile Q(p) via the residue form, Eq. (4).

    Evaluated at `dps + guard_digits` working precision, matching the
    accumulation precision used in `build_normal_contour` (see its
    docstring).
    """
    with mp.workdps(dps + guard_digits):
        return cc.residue_quantile(contour.acc, mp.mpf(p))


def normal_quantile(contour: NormalContour, p: float, mu: float = 0.0, sigma: float = 1.0,
                     dps: int = 24, guard_digits: int = 12) -> mp.mpf:
    """Affine reconstruction Q~(p) = mu + sigma * Q(p) (methodology, exp_5).

    Evaluated at `dps + guard_digits` working precision, matching the
    accumulation precision used in `build_normal_contour` (see its
    docstring).
    """
    with mp.workdps(dps + guard_digits):
        q = cc.residue_quantile(contour.acc, mp.mpf(p))
        return mp.mpf(mu) + mp.mpf(sigma) * q


def reference_quantile_normal(mu: float, sigma: float, p: float, dps: int = 30) -> mp.mpf:
    """Normal reference quantile via SciPy-seeded Newton's method in high
    precision (methodology, exp_5): `scipy.stats.norm.ppf(p)` as the
    double-precision seed, refined with `cc.newton_quantile` against the
    real CDF Phi((x-mu)/sigma) = (1+erf((x-mu)/(sigma*sqrt(2))))/2."""
    from scipy.stats import norm
    with mp.workdps(dps):
        mu_m, sigma_m = mp.mpf(mu), mp.mpf(sigma)
        x0 = float(norm.ppf(p, loc=mu, scale=sigma))

        def cdf_real(x):
            return (1 + mp.erf((mp.mpf(x) - mu_m) / (sigma_m * mp.sqrt(2)))) / 2

        return cc.newton_quantile(cdf_real, p, x0, dps=dps)


def lognormal_quantile(contour: NormalContour, p: float, sigma: float, a: float,
                        dps: int = 24, guard_digits: int = 12) -> mp.mpf:
    """Log-normal reconstruction Q~(p) = sigma * exp(a * Q(p)) (methodology,
    exp_5), reusing the SAME standard-normal contour built once for the
    plain normal case -- `sigma` is the log-normal median and `a` the
    log-scale standard deviation (this paper's convention).

    Evaluated at `dps + guard_digits` working precision, matching the
    accumulation precision used in `build_normal_contour` (see its
    docstring).
    """
    with mp.workdps(dps + guard_digits):
        q = cc.residue_quantile(contour.acc, mp.mpf(p))
        return mp.mpf(sigma) * mp.e ** (mp.mpf(a) * q)


def reference_quantile_lognormal(sigma: float, a: float, p: float, dps: int = 30) -> mp.mpf:
    """Log-normal reference quantile via SciPy-seeded Newton's method in high
    precision (methodology, exp_5): SciPy's `lognorm.ppf(p, s, scale=scale)`
    with `s = a`, `scale = sigma` matches this paper's convention, used as
    the double-precision seed; refined with `cc.newton_quantile` against the
    real CDF Phi(log(x/sigma)/a) = (1+erf(log(x/sigma)/(a*sqrt(2))))/2."""
    from scipy.stats import lognorm
    with mp.workdps(dps):
        sigma_m, a_m = mp.mpf(sigma), mp.mpf(a)
        x0 = float(lognorm.ppf(p, a, scale=sigma))

        def cdf_real(x):
            return (1 + mp.erf(mp.log(mp.mpf(x) / sigma_m) / (a_m * mp.sqrt(2)))) / 2

        return cc.newton_quantile(cdf_real, p, x0, lo=1e-300, dps=dps)
