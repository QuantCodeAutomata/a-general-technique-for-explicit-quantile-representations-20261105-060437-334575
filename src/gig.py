"""GIG (generalized inverse Gaussian) contour quantile (exp_1).

Implements the five-step general contour method of Schadner (2026) for the
GIG working density

    g(x) = x^(a-1) / (2 K_a(b)) * exp[-b/2 (x + 1/x)],   x > 0            (methodology, exp_1)

with contour height

    h(x) = x^2 / (b + 2|a-1|x + b x^2),   x > 0                           (Step 3, exp_1)

and contour derivative

    kappa'(x) = 1 + i (2 b x + 2|a-1| x^2) / (b + 2|a-1| x + b x^2)^2     (Step 5, exp_1)

The quantile is evaluated with the logarithmic contour-integral formula
(Step 5, exp_1):

    Q_GIG(p) = (1/pi) Im int_0^inf kappa'(x) Log[(P_GIG(kappa(x)) - p)/(1-p)] dx

reconstructed to scale via Q~_GIG(p; sigma, a, b) = sigma * Q_GIG(p).
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from src import contour_core as cc


def g_gig(z: mp.mpc, a: float, b: float) -> mp.mpc:
    """Holomorphic continuation of the GIG working density (Step 1, exp_1)."""
    a_m, b_m = mp.mpf(a), mp.mpf(b)
    return mp.power(z, a_m - 1) / (2 * mp.besselk(a_m, b_m)) * mp.e ** (-b_m / 2 * (z + 1 / z))


def h_gig(x: mp.mpf, a: float, b: float) -> mp.mpf:
    """Contour height (Step 3, exp_1)."""
    a_m, b_m = mp.mpf(a), mp.mpf(b)
    x = mp.mpf(x)
    return x ** 2 / (b_m + 2 * abs(a_m - 1) * x + b_m * x ** 2)


def kappa_gig(x: mp.mpf, a: float, b: float) -> mp.mpc:
    return mp.mpc(x, h_gig(x, a, b))


def kappa_prime_gig(x: mp.mpf, a: float, b: float) -> mp.mpc:
    """Analytic contour derivative (Step 5, exp_1)."""
    a_m, b_m = mp.mpf(a), mp.mpf(b)
    x = mp.mpf(x)
    denom = (b_m + 2 * abs(a_m - 1) * x + b_m * x ** 2) ** 2
    imag = (2 * b_m * x + 2 * abs(a_m - 1) * x ** 2) / denom
    return mp.mpc(1, imag)


@dataclass
class GIGContour:
    a: float
    b: float
    acc: "cc.ContourAccumulation"


def build_gig_contour(a: float, b: float, dps: int = 24, max_cells_per_side: int = 160) -> GIGContour:
    """Build the p-independent GIG contour once (Theorem 2.1 / Step 1-4, exp_1)."""
    with mp.workdps(dps):
        a_m, b_m = mp.mpf(a), mp.mpf(b)

        def h(x):
            return h_gig(x, a, b)

        def g(z):
            return g_gig(z, a, b)

        def kappa(x):
            return kappa_gig(x, a, b)

        def kappa_prime(x):
            return kappa_prime_gig(x, a, b)

        x_left = b_m / 130  # "accumulation starts ... at x = b/130 for GIG" (implementation_notes, exp_1)

        def right_order(x):
            # density orders O(x^a e^{-bx/2}) at infinity (Step 4, exp_1); absolute tail bound
            return abs(x ** a_m * mp.e ** (-b_m * x / 2))

        x_start_search = max(mp.mpf(1), x_left * 2)
        x_hi = cc.find_right_truncation(right_order, x_start_search, target_log=60.0)
        center = max(mp.mpf(1), min(x_hi / 4, mp.mpf(10)))
        acc = cc.build_contour(g, h, center, x_left, x_hi, kappa=kappa, kappa_prime=kappa_prime,
                                p_left=mp.mpf(0),
                                mesh_kwargs=dict(width_factor=mp.mpf('1.0'), growth=mp.mpf('1.08'),
                                                  max_cells_per_side=max_cells_per_side))
        return GIGContour(a=a, b=b, acc=acc)


def gig_quantile_log(contour: GIGContour, p: float, dps: int = 24) -> mp.mpf:
    """Logarithmic contour-integral formula Q_GIG(p) (Step 5, exp_1).

    Eq. (1) is the improper integral over the true open interval I=(0,infinity).
    The mesh only covers [x_lo, x_hi] with x_lo = b/130, the point below which
    the neglected *probability mass* is below e^{-60} (implementation_notes,
    exp_1). That truncation is immaterial for the residue form (Eq. 4): its
    integrand is proportional to g(kappa(x)), which vanishes super-exponentially
    as x -> 0 (the GIG density is O(x^{a-1} e^{-b/(2x)}) there). It is *not*
    immaterial for the log form: Log(ratio(x)) tends to the finite constant
    Log(-p/(1-p)) as x -> 0 (ratio(x) -> (0-p)/(1-p) since P(kappa(x)) -> 0),
    taken on the upper-half-plane branch (argument +pi), while kappa'(x) -> 1,
    so the omitted segment [0, x_lo] contributes
        int_0^{x_lo} kappa'(x) Log(ratio(x)) dx
            ~= kappa(x_lo) * Log(-p/(1-p)) - kappa(0) * Log(-p/(1-p))
            = kappa(x_lo) * [ln(p/(1-p)) + i*pi]
    (kappa(0) = 0, and P(kappa(x)) is indistinguishable from 0 to float
    precision all along this segment, since the neglected mass there is below
    e^{-60}). This is the same device the paper uses explicitly for the
    variance-gamma half-axis contours, "a straight segment from 0 to
    kappa(10^{-14})" (implementation_notes, exp_7): a numerical mesh must
    always account for the straight run from the true endpoint to the first
    resolved point, because it is not automatically negligible for every
    observable even when it is negligible for the probability mass. Adding
    this term brings Eq. (1) into agreement with the residue form Eq. (4) and
    the Newton-method reference to the same ~1e-23 relative error (verified
    in tests/test_gig.py); the two forms are identical representations of the
    quantile once both are evaluated as the stated improper integral.
    """
    with mp.workdps(dps):
        p_m = mp.mpf(p)
        acc = contour.acc
        total = mp.mpc(0)
        for kp, P, w in zip(acc.kappa_prime, acc.P_vals, acc.mesh.ws):
            ratio = (P - p_m) / (1 - p_m)
            if ratio.real <= 0 and abs(ratio.imag) < mp.mpf(10) ** (-(dps - 2)):
                # Upper-half-plane limit of the logarithm: argument +pi when the
                # imaginary part underflows near the left endpoint (implementation_notes, exp_1).
                log_val = mp.log(abs(ratio)) + mp.mpc(0, mp.pi)
            else:
                log_val = mp.log(ratio)
            total += kp * log_val * w

        # Straight-segment correction for the omitted [0, x_lo] run (see docstring).
        x_lo = acc.mesh.x_lo
        kappa_lo = kappa_gig(x_lo, contour.a, contour.b)
        ratio_left = (mp.mpf(0) - p_m) / (1 - p_m)  # P(kappa(x)) ~= 0 all along [0, x_lo]
        log_left = mp.log(abs(ratio_left)) + mp.mpc(0, mp.pi)
        total += kappa_lo * log_left

        return (mp.mpf(1) / mp.pi) * total.imag


def gig_quantile(contour: GIGContour, p: float, dps: int = 24) -> mp.mpf:
    """GIG quantile via the residue form, Eq. (4) (used generically in exp_10
    for all twelve working laws, and proven in the paper to be an equivalent
    representation to the logarithmic form of Step 5, exp_1 -- see
    Appendix B.5). This is the quantile reported for GIG in this repository."""
    with mp.workdps(dps):
        return cc.residue_quantile(contour.acc, mp.mpf(p))


def gig_quantile_scaled(contour: GIGContour, p: float, sigma: float = 1.0, dps: int = 24) -> mp.mpf:
    """Scale reconstruction Q~_GIG(p; sigma, a, b) = sigma * Q_GIG(p) (Step 5, exp_1)."""
    return mp.mpf(sigma) * gig_quantile(contour, p, dps=dps)


def reference_quantile_gig(a: float, b: float, p: float, dps: int = 30) -> mp.mpf:
    """Reference quantile via SciPy-seeded Newton's method in high precision.

    SciPy's `geninvgauss` uses the same density x^(p-1) e^{-b/2(x+1/x)} / (2 K_p(b)),
    so its shape parameter is this paper's `a` (implementation_notes, exp_1/exp_10:
    "Newton's method starts from double-precision quantiles computed with SciPy").
    """
    from scipy.stats import geninvgauss
    x0 = float(geninvgauss.ppf(p, a, b))

    def cdf_real(x):
        return cc.mp.quad(lambda t: g_gig(t, a, b), [mp.mpf(b) / 200, x])

    return cc.newton_quantile(cdf_real, p, x0, lo=1e-300, hi=None, dps=dps)
