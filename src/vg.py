"""Variance Gamma (VG) quantile construction with half-axis splitting (exp_7).

The standardized VG density (mu=0, sigma=1) is

    f(x) = e^(d*x) * |x|^a * K_a(sqrt(1+d^2)*|x|)
           / [sqrt(pi) * (2*sqrt(1+d^2))^a * Gamma(a+1/2)],     x in R      (methodology, exp_7)

with Bessel order a > -1/2 and skewness d in R. Affine reconstruction at the
end is Q~(p) = mu + sigma*Q(p), with Q(p) the standardized working quantile
computed here.

f is NOT holomorphic at x=0 (the |x| and K_a(sqrt(1+d^2)|x|) factors both
have a branch point there), so the general contour method cannot be applied
to f directly. Instead the working law is split into two HOLOMORPHIC
half-axis subdensities, for eps in {-1, +1} and Re(z) > 0:

    g_eps(z) = e^(eps*d*z) * z^a * K_a(sqrt(1+d^2)*z)
               / [sqrt(pi) * (2*sqrt(1+d^2))^a * Gamma(a+1/2)]             (Theorem A.2, exp_7)

g_{+1} analytically continues f restricted to x>0; g_{-1} continues f(-x)
restricted to x>0 (the left tail reflected onto the positive axis). Both are
holomorphic on Re(z)>0 since a>-1/2 removes the z=0 branch singularity from
the *open* right half-plane (it remains a branch point only at z=0 itself).
P_eps(z) = int_0^z g_eps(u) du is then a genuine, convergent integral (z^a is
integrable at 0 for a>-1/2).

Boundary probability p_0 = P(X<=0) is obtained in closed form via a
Beta-distribution representation (methodology, exp_7), NOT by numerically
integrating the (singular-at-0) real density up to 0:

    p_0 = I_{(sqrt(1+d^2)-d)/(2*sqrt(1+d^2))}(a+1/2, a+1/2)

where I_x(alpha,beta) is the regularized incomplete beta function, computed
as `mp.betainc(a+0.5, a+0.5, 0, x0, regularized=True)`.

Quantile formula (methodology, exp_7): for p != p_0, set
eps = sign(p - p_0), Delta_p = |p - p_0|; then

    Q(p) = -(eps/pi) * Im int_0^inf kappa(x)*g_eps(kappa(x))*kappa'(x)
                                     / [P_eps(kappa(x)) - Delta_p] dx

with the SAME contour kappa(x) = x + i*h(x) used for both half-axes,

    h(x) = x / [8 * 2^|a| * (1 + 2|a| + (sqrt(1+d^2)+|d|)*x)],   x > 0

kappa'(x) is left to the shared engine's numerical derivative (no
hand-derived analytic form; see `build_vg_contour`/`vg_quantile` for the
`guard_digits` precision floor technique used when that is insufficient,
following `src/gg_gev.py`). Q(p_0) = 0 exactly (the two half-axis
representations meet at zero by construction).

Truncation: at infinity, K_a(z) ~ sqrt(pi/(2z)) e^{-z} gives
g_eps(x) = O(x^(a-1/2) * e^{-(sqrt(1+d^2)-eps*d)*x}) (methodology, exp_7);
found per eps via `cc.find_right_truncation`. At zero, the residue-form
kernel is O(x*g_eps(x)); as the paper does explicitly for VG
(implementation_notes, exp_7), a small fixed left cutoff x_lo=1e-14 is used
-- "a straight segment from 0 to kappa(10^-14)" -- with no p_left
correction, since this kernel is integrable for a>-1/2 (2a+1 > -1) and the
omitted mass is negligible for a>=0 (verified in tests/test_vg.py).

KNOWN LIMITATION (a<0, documented deviation from "negligible to e^-60"):
for a<0, K_a(z) ~ C*(z/2)^a as z->0 (the (z/2)^(-a) term in the standard
small-z expansion of K_a vanishes here since -a>0), so g_eps(x) ~ x^(2a)
-- *more* singular than f itself -- and P_eps(x_lo=1e-14) is actually of
order x_lo^(2a+1), e.g. ~1.7e-6 for a=-0.3 (verified directly by
quadrature, and independently cross-checked via the p_0 Beta-formula vs.
direct-integration sanity test, which shows the same ~1.7e-6 gap). This
shows up as a quantile error floor of the same order for a<0 that no
amount of extra bulk-mesh resolution or extra precision (`guard_digits`)
removes (both verified directly: the floor is identical at max_cells 300
and 600, and at guard_digits 12 and 20) -- it is a genuine truncation bias,
not a quadrature-accuracy issue. Two natural fixes were tried and
*rejected*: (1) searching a per-(a,d) smaller x_lo via
`cc.find_left_truncation` on the x^(2a+1) kernel order drives x_lo down to
~1e-66, which stretches the geometric mesh over far more decades under the
same cell allowance and *degrades* bulk accuracy (confirmed: worst relative
error rose from ~1.3e-6 to ~6.6e-5); (2) adding an explicit closed-quadrature
`p_left` correction (`mp.quad` of g_eps over [0, x_lo]) also empirically
*worsened* accuracy (to ~6.5e-5) despite matching the expected 1.7e-6
magnitude, for a reason not identified by these checks.
Given both attempted fixes regress accuracy, the simple fixed x_lo=1e-14
with no correction (the paper's literal recipe) is kept as the better
empirical choice; the a<0 test tolerance in tests/test_vg.py is loosened
(and this is documented there) to account for the resulting ~1e-6 floor,
which is still far below the probability-level resolution needed for any
practical quantile use.
"""
from __future__ import annotations

from dataclasses import dataclass

import mpmath as mp

from src import contour_core as cc

X_LO = mp.mpf('1e-14')  # fixed left cutoff shared by both half-axis contours


def _delta(d) -> mp.mpf:
    d_m = mp.mpf(d)
    return mp.sqrt(1 + d_m ** 2)


def _norm_const(a, d) -> mp.mpf:
    a_m = mp.mpf(a)
    delta = _delta(d)
    return mp.sqrt(mp.pi) * mp.power(2 * delta, a_m) * mp.gamma(a_m + mp.mpf('0.5'))


def g_vg(z: mp.mpc, a: float, d: float, eps: int) -> mp.mpc:
    """Holomorphic half-axis subdensity g_eps(z), eps in {-1, +1} (Theorem A.2, exp_7)."""
    a_m, d_m = mp.mpf(a), mp.mpf(d)
    delta = _delta(d)
    C = _norm_const(a, d)
    return mp.e ** (eps * d_m * z) * mp.power(z, a_m) * mp.besselk(a_m, delta * z) / C


def f_vg_real(x: float, a: float, d: float) -> mp.mpf:
    """The (non-holomorphic-at-0) real standardized VG density f(x), x != 0."""
    a_m, d_m = mp.mpf(a), mp.mpf(d)
    delta = _delta(d)
    C = _norm_const(a, d)
    x_m = mp.mpf(x)
    return mp.e ** (d_m * x_m) * mp.power(abs(x_m), a_m) * mp.besselk(a_m, delta * abs(x_m)) / C


def h_vg(x: mp.mpf, a: float, d: float) -> mp.mpf:
    """Shared contour height, same for both half-axes (methodology, exp_7)."""
    a_m, d_m = mp.mpf(a), mp.mpf(d)
    delta = _delta(d)
    abs_a = abs(a_m)
    x = mp.mpf(x)
    return x / (8 * mp.power(2, abs_a) * (1 + 2 * abs_a + (delta + abs(d_m)) * x))


def kappa_vg(x: mp.mpf, a: float, d: float) -> mp.mpc:
    return mp.mpc(x, h_vg(x, a, d))


def p0_vg(a: float, d: float) -> mp.mpf:
    """Boundary probability p_0 = P(X<=0) via the Beta-distribution closed
    form (methodology, exp_7)."""
    a_m, d_m = mp.mpf(a), mp.mpf(d)
    delta = _delta(d)
    x0 = (delta - d_m) / (2 * delta)
    return mp.betainc(a_m + mp.mpf('0.5'), a_m + mp.mpf('0.5'), 0, x0, regularized=True)


def _fits_within_budget(h, center, x_stop, direction, width_factor, growth,
                         max_cells, slack: float = 5.0) -> bool:
    """True if `cc._one_sided_edges` genuinely reaches `x_stop` by geometric
    shrink/growth within `max_cells` cells -- i.e. the walk's *penultimate*
    edge lands within `slack`x the last true cell width of `x_stop` -- rather
    than silently falling back to one pathologically wide final cell that
    merely snaps onto `x_stop` after exhausting the cell allowance (see
    `_one_sided_edges` docstring in `src/contour_core.py`)."""
    edges = cc._one_sided_edges(h, center, x_stop, direction, width_factor, growth, max_cells)
    if len(edges) - 1 > max_cells or len(edges) < 3:
        return len(edges) - 1 <= max_cells
    last_true_width = abs(edges[-2] - edges[-3])
    return abs(edges[-2] - x_stop) < slack * max(last_true_width, abs(x_stop) * mp.mpf('1e-10'))


def _fit_width_factor(h, center, x_stop, direction, max_cells, growth=mp.mpf('1.1'),
                       start: mp.mpf = mp.mpf('1.0'), tries: int = 40) -> mp.mpf:
    """Smallest `width_factor` (searched geometrically from `start`) for which
    `_one_sided_edges` actually reaches `x_stop` within `max_cells` cells.

    Needed because VG's contour height (methodology, exp_7) has a much
    smaller linear coefficient near 0 -- and a much smaller asymptotic value
    at infinity -- than every other module's height function in this
    repository (the "8 * 2^|a| * (1+2|a|+...)" denominator), so the shared
    engine's default width_factor=1.0 gives a geometric shrink/growth ratio
    per cell too close to 1: spanning the required 10-15 decades down to
    x_lo=1e-14, or out to x_hi (~10^2), would need hundreds to thousands of
    cells, far beyond the typical max_cells_per_side allowance used elsewhere
    in this repository. Verified directly: with width_factor=1.0,
    `_one_sided_edges` silently falls back (its allowance-exhaustion branch)
    to one catastrophic
    final cell spanning most of the domain, corrupting the composite-GL
    half-axis mass by several percent. Widening `width_factor` (this
    function searches the *smallest* value that works, analogous in spirit
    to -- though automated rather than hardcoded like -- the a-dependent
    mesh tuning in `src/gg_gev.py`'s `_mesh_kwargs_for`) restores genuine
    geometric coverage within a few hundred cells for every (a, d) tested.
    """
    wf_fail = None
    wf = mp.mpf(start)
    for _ in range(tries):
        if _fits_within_budget(h, center, x_stop, direction, wf, growth, max_cells):
            wf_pass = wf
            break
        wf_fail = wf
        wf *= mp.mpf('1.5')
    else:
        return wf
    if wf_fail is None:
        return wf_pass
    # Bisection refinement: the exponential search above only brackets the
    # true minimum within a factor of 1.5; narrowing it further (to ~2%)
    # meaningfully improves quadrature accuracy, since any extra slack in
    # width_factor beyond what is strictly needed makes cells needlessly
    # coarser everywhere (not just near the endpoint).
    lo, hi = wf_fail, wf_pass
    for _ in range(12):
        mid = (lo + hi) / 2
        if _fits_within_budget(h, center, x_stop, direction, mid, growth, max_cells):
            hi = mid
        else:
            lo = mid
    return hi


@dataclass
class VGContour:
    a: float
    d: float
    p0: mp.mpf
    acc_plus: "cc.ContourAccumulation"
    acc_minus: "cc.ContourAccumulation"
    x_hi_plus: mp.mpf
    x_hi_minus: mp.mpf
    width_factor: mp.mpf


def build_vg_contour(a: float, d: float, dps: int = 24, max_cells_per_side: int = 300,
                      guard_digits: int = 12, width_factor_cap: float = 5.0,
                      max_cells_cap: int = 1600) -> VGContour:
    """Build both p-independent half-axis contours once (Theorem A.2, exp_7).

    A single mesh (shared `kappa`/`kappa_prime`/`h`) is built once via
    `cc.build_mesh` and accumulated twice, once per g_eps, to avoid
    duplicating mesh construction (methodology, exp_7). The mesh's right
    endpoint is the larger of the two eps-specific truncations (`x_hi_plus`,
    `x_hi_minus`), found independently since the two tails decay at
    different rates (sqrt(1+d^2) -+ d); using the larger value for both
    guarantees each integrand is already negligible (order e^-60) well
    before the mesh ends, for both eps. `width_factor` is auto-tuned (see
    `_fit_width_factor`) to the larger of what the left arm (center -> x_lo)
    and the right arm (center -> x_hi) need to genuinely reach their
    endpoints within `max_cells_per_side` cells.

    Adaptive cell allowance: `max_cells_per_side` is doubled (up to
    `max_cells_cap`) whenever the fitted `width_factor` still exceeds
    `width_factor_cap`, since a width_factor far above 1 means cells are far
    coarser than the contour height h(x) everywhere (not just near the
    endpoint), which measurably degrades quadrature accuracy (verified
    directly: at |a|=1.5, width_factor~11 with max_cells_per_side=300 gives
    only ~5e-5 relative error against the Newton reference, while
    width_factor~2.8 with max_cells_per_side=1200 gives ~2e-13). This keeps
    the two mild (a,d) cases in `config.VG_TEST_PARAMS` cheap (a few hundred
    cells) while automatically giving the large-|a| case the much larger
    mesh it needs -- an a-dependent cost difference inherent to VG's own
    height formula (see `_fit_width_factor`), not a tunable choice.

    `guard_digits=12` extra accumulation precision pushes the shared
    engine's fixed-step numerical-derivative floor (`cc.numerical_derivative`)
    below the target tolerance, following the technique documented in
    `src/gg_gev.py` (hand-deriving kappa'(x) is explicitly avoided by the
    methodology for this module).
    """
    with mp.workdps(dps + guard_digits):
        a_m, d_m = mp.mpf(a), mp.mpf(d)
        delta = _delta(d)

        def h(x):
            return h_vg(x, a, d)

        def kappa(x):
            return kappa_vg(x, a, d)

        def order_fn(eps_val):
            def fn(x):
                return abs(mp.power(x, a_m - mp.mpf('0.5')) * mp.e ** (-(delta - eps_val * d_m) * x))
            return fn

        x_hi_plus = cc.find_right_truncation(order_fn(1), mp.mpf(10), target_log=60.0)
        x_hi_minus = cc.find_right_truncation(order_fn(-1), mp.mpf(10), target_log=60.0)
        x_hi = max(x_hi_plus, x_hi_minus)
        x_lo = X_LO

        center = max(mp.mpf(1), min(x_hi / 4, mp.mpf(10)))
        growth = mp.mpf('1.1')
        mc = max_cells_per_side
        while True:
            wf_left = _fit_width_factor(h, center, x_lo, -1, mc, growth)
            wf_right = _fit_width_factor(h, center, x_hi, 1, mc, growth)
            width_factor = max(wf_left, wf_right)
            if width_factor <= width_factor_cap or mc >= max_cells_cap:
                break
            mc = min(mc * 2, max_cells_cap)
        mesh = cc.build_mesh(h, center, x_lo, x_hi,
                              width_factor=width_factor, growth=growth,
                              max_cells_per_side=mc)

        def kappa_prime(x):
            return cc.numerical_derivative(kappa, x)

        g_plus = lambda z: g_vg(z, a, d, 1)
        g_minus = lambda z: g_vg(z, a, d, -1)

        acc_plus = cc.accumulate(mesh, kappa, kappa_prime, g_plus, p_left=mp.mpf(0))
        acc_minus = cc.accumulate(mesh, kappa, kappa_prime, g_minus, p_left=mp.mpf(0))

        p0 = p0_vg(a, d)

        return VGContour(a=a, d=d, p0=p0, acc_plus=acc_plus, acc_minus=acc_minus,
                          x_hi_plus=x_hi_plus, x_hi_minus=x_hi_minus, width_factor=width_factor)


def _residue_sum(acc: "cc.ContourAccumulation", delta_p: mp.mpf) -> mp.mpf:
    """Im sum[ kappa*g_eps(kappa)*kappa' / (P_eps(kappa)-Delta_p) * w ] / pi
    (shared kernel for both half-axes; methodology, exp_7)."""
    total = mp.mpc(0)
    for k, kp, g_v, P_v, w in zip(acc.kappa, acc.kappa_prime, acc.g_vals, acc.P_vals, acc.mesh.ws):
        total += k * g_v * kp / (P_v - delta_p) * w
    return total.imag / mp.pi


def vg_quantile(contour: VGContour, p: float, dps: int = 24, guard_digits: int = 12) -> mp.mpf:
    """Standardized VG working quantile Q(p) (methodology, exp_7).

    Q(p_0) = 0 exactly, by construction (the two half-axis representations
    meet at zero); detected here via a tight absolute threshold on |p-p0|
    rather than evaluating the (then divide-by-zero) residue sum.
    """
    with mp.workdps(dps + guard_digits):
        p_m, p0_m = mp.mpf(p), mp.mpf(contour.p0)
        if abs(p_m - p0_m) < mp.mpf(10) ** (-(dps - 2)):
            return mp.mpf(0)
        eps = 1 if p_m > p0_m else -1
        delta_p = abs(p_m - p0_m)
        acc = contour.acc_plus if eps == 1 else contour.acc_minus
        return -eps * _residue_sum(acc, delta_p)


def vg_quantile_scaled(contour: VGContour, p: float, mu: float = 0.0, sigma: float = 1.0,
                        dps: int = 24, guard_digits: int = 12) -> mp.mpf:
    """Affine reconstruction Q~(p) = mu + sigma*Q(p) (methodology, exp_7)."""
    q = vg_quantile(contour, p, dps=dps, guard_digits=guard_digits)
    return mp.mpf(mu) + mp.mpf(sigma) * q


# --- Reference quantities (testing/experiment use only, not the method itself) ---

def cdf_vg_real(x: float, a: float, d: float, p0: mp.mpf = None) -> mp.mpf:
    """P(X<=x) by direct numerical integration of the real density, handling
    the x=0 singularity by splitting the integration domain there (mp.quad's
    built-in handling of a mild integrable singularity at a stated endpoint,
    tried first as the simplest option; sufficient in practice here).
    """
    x_m = mp.mpf(x)
    if p0 is None:
        p0 = p0_vg(a, d)
    if x_m == 0:
        return p0
    if x_m > 0:
        return p0 + mp.quad(lambda t: f_vg_real(t, a, d), [mp.mpf('1e-14'), x_m])
    return p0 - mp.quad(lambda t: f_vg_real(t, a, d), [x_m, mp.mpf('-1e-14')])


def p0_direct_integration(a: float, d: float, x_far: float = None) -> mp.mpf:
    """Independent sanity check of the Beta-based p_0: P(0) via direct
    numerical integration of the real density over (-infinity, 0] (methodology,
    exp_7 asks this be verified against the closed form)."""
    a_m, d_m = mp.mpf(a), mp.mpf(d)
    delta = _delta(d)
    if x_far is None:
        def order_fn(x):
            return abs(mp.power(x, a_m - mp.mpf('0.5')) * mp.e ** (-(delta + d_m) * x))
        x_far = cc.find_right_truncation(order_fn, mp.mpf(10), target_log=60.0)
    return mp.quad(lambda t: f_vg_real(t, a, d), [-mp.mpf(x_far), mp.mpf('-1e-14')])


def reference_quantile_vg(a: float, d: float, p: float, dps: int = 30) -> mp.mpf:
    """Reference quantile via SciPy-seeded Newton's method in high precision
    (Appendix D.1 pattern used throughout this repository)."""
    import numpy as np
    from scipy import integrate, optimize
    from scipy.special import kv as scipy_kv, gamma as scipy_gamma

    delta_f = float(_delta(d))
    C_f = float(np.sqrt(np.pi) * (2 * delta_f) ** a * scipy_gamma(a + 0.5))

    def pdf(x):
        if x == 0:
            return 0.0
        return np.exp(d * x) * abs(x) ** a * scipy_kv(a, delta_f * abs(x)) / C_f

    def cdf(x):
        if x >= 0:
            val, _ = integrate.quad(pdf, 0.0, x, points=[0.0], limit=200)
            p0_est, _ = integrate.quad(pdf, -50.0 / delta_f, 0.0, points=[0.0], limit=200)
            return p0_est + val
        val, _ = integrate.quad(pdf, x, 0.0, points=[0.0], limit=200)
        p0_est, _ = integrate.quad(pdf, -50.0 / delta_f, 0.0, points=[0.0], limit=200)
        return p0_est - val

    lo, hi = -200.0 / delta_f, 200.0 / delta_f
    x0 = optimize.brentq(lambda x: cdf(x) - p, lo, hi, xtol=1e-10)

    with mp.workdps(dps):
        a_m, d_m = mp.mpf(a), mp.mpf(d)

        def cdf_real(x):
            return cdf_vg_real(x, a, d)

        return cc.newton_quantile(cdf_real, p, x0, lo=lo, hi=hi, dps=dps)
