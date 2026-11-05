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

    Implemented exactly as specified, with the upper-half-plane limit of the
    logarithm (argument +pi) applied whenever the imaginary part underflows
    near the left endpoint (implementation_notes, exp_1).

    Integration by parts gives int kappa' Log(ratio) dx = [kappa Log(ratio)]
    - int kappa g(kappa) kappa' / (P(kappa)-p) dx, so Eq. (1) and the residue
    form Eq. (4) must coincide once both the x->0 and x->infinity boundary
    terms vanish. kappa(x_hi)*Log(ratio(x_hi)) is checked to vanish (the
    right tail decays as the paper's Step 4 orders predict). Numerically,
    this implementation's Eq. (1) value differs from the reference quantile
    by a few percent, and that gap is invariant under both mesh refinement
    (657 vs 160 cells per side) and shrinking the left truncation point by
    twelve orders of magnitude (x_left from b/130 down to 1e-18): a true
    quadrature or left-boundary-term artifact would shrink with either
    change, so neither is the cause. The discrepancy is therefore recorded
    as an unresolved implementation gap of this pointwise-principal-branch
    evaluation of Eq. (1) (see "Deviations from the paper", results/RESULTS.md),
    not attributed to the quadrature mesh. `gig_quantile` (the residue form,
    Eq. 4) matches the Newton-method reference to about 1e-23 and is used
    for every reported GIG number in this repository, including exp_10.
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
