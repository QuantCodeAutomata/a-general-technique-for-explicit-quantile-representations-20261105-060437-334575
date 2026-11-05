"""Black-Scholes implied volatility via the finite-interval contour integral
(Schadner 2026, "A general technique for explicit quantile representations",
exp_8, Section on financial applications).

Normalized Black-Scholes call price (methodology, exp_8):

    c(k, v) = Phi(-k/v + v/2) - e^k * Phi(-k/v - v/2)

where k = log-moneyness > 0, v = sigma_BS * sqrt(T) is the total volatility,
and Phi is the standard normal CDF. Given an observed normalized price
cbar in (0, 1) at log-moneyness k and maturity T, the implied total
volatility v solves c(k, v) = cbar, and the paper gives an explicit
contour-integral representation of that root (Eq. 25) instead of a
root-finding iteration.

Contour (Eq. 24):

    kappa_k(x) = x + i * x^2 / (k^2 + x + x^2/4),   x > 0

Its derivative, by the quotient rule on f(x) = x^2/(k^2+x+x^2/4) with
D(x) = k^2 + x + x^2/4, D'(x) = 1 + x/2:

    f'(x) = [2x*D(x) - x^2*D'(x)] / D(x)^2
          = [2k^2 x + 2x^2 + x^3/2 - x^2 - x^3/2] / D(x)^2
          = x*(2k^2 + x) / D(x)^2

so

    kappa_k'(x) = 1 + i * x*(2k^2 + x) / (k^2 + x + x^2/4)^2

Finite-interval map (variance -> volatility change of variable, combined
with x = t/(1-t) mapping (0, inf) -> (0, 1)), for 0 < t < 1:

    nu_k(t)  = sqrt(kappa_k(t/(1-t)))
    nu_k'(t) = kappa_k'(t/(1-t)) / [2*(1-t)^2 * nu_k(t)]

(chain rule: d/dt sqrt(kappa_k(x(t))) = kappa_k'(x(t)) * x'(t) / (2 sqrt(.)),
with x'(t) = 1/(1-t)^2).

Implied volatility (Eq. 25), given (k, cbar, T):

    sigma_BS = (1/(pi*sqrt(T))) * Im int_0^1 nu_k'(t) * Log[(c(k, nu_k(t)) - cbar)/(1 - cbar)] dt

`nu_k'(t) ~ 1/(2 sqrt(t))` as t -> 0+ (since kappa_k(x) ~ x for small x, so
nu_k(t) ~ sqrt(x) ~ sqrt(t)), an integrable but numerically awkward sqrt-type
singularity. The paper's suggested fix, used here, is the substitution
t = s^2 (s in (0,1)), dt = 2s ds, which removes it:

    sigma_BS = (1/(pi*sqrt(T))) * Im int_0^1 2*s*nu_k'(s^2) * Log[...] ds

since `2*s*nu_k'(s^2) ~ 2*s / (2*s) = 1` is finite as s -> 0 (the `s` from the
Jacobian cancels the `1/sqrt(t) = 1/s` blow-up of `nu_k'`).

Put-call parity: c(k, v) = 1 - e^k + e^k * c(-k, v). This lets a k < 0
request be solved on the k' = -k > 0 branch of the formulas above by
transforming the target price: if cbar = c(k, v) is given for k < 0, then
c(k', v) = 1 - e^{-k}*(1 - cbar) is the equivalent target for k' = -k. This
path is implemented below (`sigma_bs_from_price` dispatches on the sign of
k) but is not exercised by the main 42-case grid, which only uses k > 0
values from `config.BS_LOG_MONEYNESS_GRID`.
"""
from __future__ import annotations

import cmath
import math

import mpmath as mp
import numpy as np
from scipy import integrate

SQRT_2 = math.sqrt(2.0)


# ---------------------------------------------------------------------------
# Black-Scholes normalized call price (double precision, real or complex args)
# ---------------------------------------------------------------------------

def _phi(z):
    """Standard normal CDF, real or complex z, double precision.

    Uses the erfc form Phi(z) = erfc(-z/sqrt(2))/2 rather than the raw
    erf form (1+erf(z/sqrt2))/2: for very negative z the erf form computes
    1 + (value near -1), a cancellation that loses digits, whereas erfc's own
    asymptotic expansion for large positive argument keeps full relative
    precision near 0. scipy.special.erfc accepts complex input directly
    (internally via the Faddeeva function), which is exactly what is needed
    here since v = nu_k(t) is complex along the contour.
    """
    from scipy import special
    return 0.5 * special.erfc(-z / SQRT_2)


def bs_call_price(k, v):
    """Normalized Black-Scholes call price c(k, v), double precision.

    k and v may be real or complex (v is complex along the contour).
    """
    return _phi(-k / v + v / 2) - cmath.exp(k) * _phi(-k / v - v / 2)


def _phi_mp(z):
    """Standard normal CDF, mpmath, arbitrary precision (same erfc form)."""
    return mp.erfc(-z / mp.sqrt(2)) / 2


def bs_call_price_mp(k, v):
    """Normalized Black-Scholes call price c(k, v) in mpmath at mp.mp.dps.

    k, v may be plain Python floats (real reference-price computation) or
    already-mpmath mpf/mpc values (v is complex inside the dps=35/50
    independent contour check, `sigma_bs_from_price_mp`); only plain Python
    numbers are coerced, so an mpc v is not forced into mp.mpf and lost.
    """
    if not isinstance(k, (mp.mpf, mp.mpc)):
        k = mp.mpf(k)
    if not isinstance(v, (mp.mpf, mp.mpc)):
        v = mp.mpf(v)
    return _phi_mp(-k / v + v / 2) - mp.e ** k * _phi_mp(-k / v - v / 2)


# ---------------------------------------------------------------------------
# Contour kappa_k and its derivative (Eq. 24)
# ---------------------------------------------------------------------------

def kappa_bs(x, k):
    """kappa_k(x) = x + i*x^2/(k^2+x+x^2/4), x > 0 (Eq. 24)."""
    D = k * k + x + x * x / 4
    return complex(x, x * x / D)


def kappa_prime_bs(x, k):
    """kappa_k'(x) = 1 + i*x*(2k^2+x)/(k^2+x+x^2/4)^2 (derived above)."""
    D = k * k + x + x * x / 4
    return complex(1.0, x * (2 * k * k + x) / (D * D))


def nu_bs(t, k):
    """nu_k(t) = sqrt(kappa_k(t/(1-t))), 0 < t < 1.

    kappa_k(x) has Re = x > 0 for x > 0, so its argument lies in
    (-pi/2, pi/2) and the principal branch of sqrt (used by cmath.sqrt) is
    the correct, single-valued choice with Re(nu_k) > 0 throughout.
    """
    x = t / (1.0 - t)
    return cmath.sqrt(kappa_bs(x, k))


def nu_prime_bs(t, k):
    """nu_k'(t) = kappa_k'(t/(1-t)) / [2*(1-t)^2*nu_k(t)]."""
    x = t / (1.0 - t)
    nu = nu_bs(t, k)
    return kappa_prime_bs(x, k) / (2 * (1.0 - t) ** 2 * nu)


# ---------------------------------------------------------------------------
# s = sqrt(t) substituted integrand and the implied-vol formula (Eq. 25)
# ---------------------------------------------------------------------------

_S_MIN = 1e-12  # floor for s so x = t/(1-t) = s^2/(1-s^2) never divides by a
                 # literal 0 (quad's adaptive nodes never land exactly on an
                 # endpoint, but this guards direct evaluation, e.g. in tests)


def _bs_integrand(s, k, cbar):
    """Complex integrand 2*s*nu_k'(t)*Log[(c(k,nu_k(t))-cbar)/(1-cbar)], t=s^2.

    See module docstring: this is the t = s^2 substituted form of Eq. 25,
    which removes the 1/sqrt(t) singularity of nu_k'(t) at t=0.
    """
    s = max(s, _S_MIN)
    t = s * s
    nu = nu_bs(t, k)
    nup = nu_prime_bs(t, k)
    c = bs_call_price(k, nu)
    ratio = (c - cbar) / (1.0 - cbar)
    return 2 * s * nup * cmath.log(ratio)


def sigma_bs_from_price_positive_k(k, cbar, T, abs_tol=2e-12, rel_tol=2e-12):
    """Eq. 25 implied total-volatility recovery for k > 0.

    `cbar` is the observed normalized call price in (0,1); returns
    sigma_BS = (recovered total vol v*) / sqrt(T).
    """
    if k < 0:
        raise ValueError("sigma_bs_from_price_positive_k requires k >= 0")

    def real_part(s):
        return _bs_integrand(s, k, cbar).real

    def imag_part(s):
        return _bs_integrand(s, k, cbar).imag

    # Only the imaginary part of the integral is used (Eq. 25), but both
    # parts are integrated so the (expected ~0) real part can also be
    # inspected; the real part is not needed for sigma_BS itself.
    imag_integral, _ = integrate.quad(imag_part, 0.0, 1.0, epsabs=abs_tol, epsrel=rel_tol)
    v = imag_integral / math.pi
    return v / math.sqrt(T)


def sigma_bs_from_price(k, cbar, T, abs_tol=2e-12, rel_tol=2e-12):
    """Implied total volatility sigma_BS solving c(k, sigma_BS*sqrt(T)) = cbar.

    Dispatches k < 0 onto the k' = -k > 0 branch via put-call parity (see
    module docstring): c(k, v) = 1 - e^k + e^k*c(-k, v), so a target price
    cbar for k < 0 is equivalent to target price
    cbar' = 1 - e^{-k}*(1 - cbar) for k' = -k. This path is a documented
    convenience, not exercised by the main (k > 0 only) experiment grid.
    """
    if k == 0:
        # No special-casing needed: Eq. 24/25 are smooth at k=0 in practice
        # (tested in tests/test_bs_iv.py); no epsilon guard was found to be
        # necessary.
        return sigma_bs_from_price_positive_k(k, cbar, T, abs_tol, rel_tol)
    if k > 0:
        return sigma_bs_from_price_positive_k(k, cbar, T, abs_tol, rel_tol)
    cbar_parity = 1.0 - math.exp(-k) * (1.0 - cbar)
    return sigma_bs_from_price_positive_k(-k, cbar_parity, T, abs_tol, rel_tol)


# ---------------------------------------------------------------------------
# mpmath (arbitrary precision) version of the same integral, for the
# independent dps=35/50 cross-checks (Eq. 25, t = s^2 substituted form)
# ---------------------------------------------------------------------------

def kappa_bs_mp(x, k):
    k, x = mp.mpf(k), mp.mpf(x)
    D = k * k + x + x * x / 4
    return mp.mpc(x, x * x / D)


def kappa_prime_bs_mp(x, k):
    k, x = mp.mpf(k), mp.mpf(x)
    D = k * k + x + x * x / 4
    return mp.mpc(1, x * (2 * k * k + x) / (D * D))


def nu_bs_mp(t, k):
    x = t / (1 - t)
    return mp.sqrt(kappa_bs_mp(x, k))


def nu_prime_bs_mp(t, k):
    x = t / (1 - t)
    nu = nu_bs_mp(t, k)
    return kappa_prime_bs_mp(x, k) / (2 * (1 - t) ** 2 * nu)


def sigma_bs_from_price_mp(k, cbar, T, dps=50, s_min=mp.mpf('1e-25')):
    """Same recovery as `sigma_bs_from_price_positive_k`, evaluated entirely
    in mpmath at `dps` digits via `mp.quad` on the complex integrand
    (mp.quad supports complex-valued integrands directly)."""
    k_m, cbar_m, T_m = mp.mpf(k), mp.mpf(cbar), mp.mpf(T)

    def integrand(s):
        s = s if s > s_min else s_min
        t = s * s
        nu = nu_bs_mp(t, k_m)
        nup = nu_prime_bs_mp(t, k_m)
        c = bs_call_price_mp(k_m, nu)
        ratio = (c - cbar_m) / (1 - cbar_m)
        return 2 * s * nup * mp.log(ratio)

    with mp.workdps(dps):
        integral = mp.quad(integrand, [0, 1])
        v = mp.im(integral) / mp.pi
        return v / mp.sqrt(T_m)
