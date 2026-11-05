"""Bachelier (normal) implied volatility via the contour integral
(Schadner 2026, "A general technique for explicit quantile representations",
exp_9, Section on financial applications).

Normalized Bachelier time-value function, v > 0:

    V(v) = v*phi(1/v) - Phi(-1/v)

with phi(z) = e^(-z^2/2)/sqrt(2*pi) the standard normal pdf and
Phi(z) = (1+erf(z/sqrt(2)))/2 the standard normal CDF. Given a Bachelier
(normal) model with forward F, strike K, maturity T and normal volatility
sigma_N, write m = |F-K| and v_N = sigma_N*sqrt(T)/m: the call's time value
(price minus intrinsic value), divided by m, equals V(v_N), i.e. V(v) is the
m-normalized time value as a function of the m-normalized volatility v.
`Vbar` below is always this normalized time value.

Equivalently (cross-checked in tests/test_bachelier_iv.py), using the upper
incomplete gamma function:

    V(v) = Gamma(-1/2, 1/(2v^2)) / (4*sqrt(pi))

Contour (Eq. E.2):

    kappa_N(x) = x + i*x^3/(2+x^3),   x > 0
    kappa_N'(x) = 1 + i*6*x^2/(2+x^3)^2   (given explicitly in the paper;
                                            quotient rule on x^3/(2+x^3)
                                            confirms: d/dx[x^3/(2+x^3)]
                                            = [3x^2(2+x^3)-x^3*3x^2]/(2+x^3)^2
                                            = 6x^2/(2+x^3)^2)

Residue form (Eq. E.3) -- note the nonvanishing closing-segment contribution
of the constant `1` (the contour here does not approach the real axis at
infinity the way the GIG/Beta/etc. contours in this repository do, so that
segment's contribution does not vanish and must be kept):

    sigma_N = (m/(pi*sqrt(T))) * [1 - int_0^inf Im[kappa_N(x)*kappa_N'(x)*phi(1/kappa_N(x))
                                                     / (V(kappa_N(x)) - Vbar)] dx]

Logarithmic finite-interval form (Eq. E.4), using x = t/(1-t) (0<t<1, so
t = x/(1+x)) and, by the chain rule (dx/dt = 1/(1-t)^2),

    nu(t)  = kappa_N(t/(1-t))
    nu'(t) = kappa_N'(t/(1-t)) / (1-t)^2

    sigma_N = (m/(pi*sqrt(T))) * int_0^1 Im[nu'(t) * Log[(V(nu(t))-Vbar)/(V(nu(t))+Vbar)]] dt

Both forms are implemented directly with double-precision scipy quadrature
(no sqrt-type substitution is needed here, unlike exp_8: kappa_N(0) = 0 and
V(kappa_N(x)) -> 0 as x -> 0+ together make both integrands finite, not
singular, at their left endpoint -- verified numerically in
tests/test_bachelier_iv.py).

At-the-money closed form: as m -> 0 (F = K), the Bachelier call price
reduces to C = sigma_N*sqrt(T)*phi(0) = sigma_N*sqrt(T)/sqrt(2*pi), so

    sigma_N = sqrt(2*pi/T) * C / D

where C is the (un-normalized) ATM call price and D is a discount factor
(D=1 if C is already expressed in forward/undiscounted units).
"""
from __future__ import annotations

import cmath
import math

import mpmath as mp
import numpy as np
from scipy import integrate, special

SQRT_2PI = math.sqrt(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)

_X_MIN = 1e-12  # floor for x so 1/kappa_N(x) never divides by a literal 0


# ---------------------------------------------------------------------------
# Normal pdf/cdf and the normalized Bachelier time-value function V(v)
# ---------------------------------------------------------------------------

def phi_pdf(z):
    """Standard normal pdf, real or complex z, double precision."""
    return cmath.exp(-z * z / 2) / SQRT_2PI


def phi_cdf(z):
    """Standard normal CDF via erfc (see src.bs_iv._phi for the same
    numerical-stability rationale); real or complex z, double precision."""
    return 0.5 * special.erfc(-z / SQRT_2)


def bachelier_time_value(v):
    """V(v) = v*phi(1/v) - Phi(-1/v); v real > 0 or complex (along the contour)."""
    return v * phi_pdf(1.0 / v) - phi_cdf(-1.0 / v)


def bachelier_time_value_gammainc(v):
    """Cross-check form V(v) = Gamma(-1/2, 1/(2v^2)) / (4*sqrt(pi)) (mpmath,
    real v > 0 only -- used only as a sanity check in tests, since it has its
    own numerical issues near v=0 where the incomplete-gamma argument blows
    up)."""
    v = mp.mpf(v)
    return mp.gammainc(mp.mpf('-0.5'), 1 / (2 * v ** 2)) / (4 * mp.sqrt(mp.pi))


# ---------------------------------------------------------------------------
# Contour kappa_N and its derivative (Eq. E.2)
# ---------------------------------------------------------------------------

def kappa_n(x):
    return complex(x, x ** 3 / (2 + x ** 3))


def kappa_n_prime(x):
    return complex(1.0, 6 * x ** 2 / (2 + x ** 3) ** 2)


def nu_n(t):
    """nu(t) = kappa_N(t/(1-t)), 0 < t < 1."""
    x = t / (1.0 - t)
    return kappa_n(x)


def nu_n_prime(t):
    """nu'(t) = kappa_N'(t/(1-t)) / (1-t)^2 (chain rule, x'(t) = 1/(1-t)^2)."""
    x = t / (1.0 - t)
    return kappa_n_prime(x) / (1.0 - t) ** 2


# ---------------------------------------------------------------------------
# Residue form (Eq. E.3) and logarithmic form (Eq. E.4)
# ---------------------------------------------------------------------------

def bachelier_iv_residue(m, Vbar, T, abs_tol=1e-12, rel_tol=1e-12):
    """sigma_N via the residue form (Eq. E.3)."""

    def imag_integrand(x):
        x = max(x, _X_MIN)
        k = kappa_n(x)
        kp = kappa_n_prime(x)
        val = k * kp * phi_pdf(1.0 / k) / (bachelier_time_value(k) - Vbar)
        return val.imag

    integral, _ = integrate.quad(imag_integrand, 0.0, np.inf,
                                  epsabs=abs_tol, epsrel=rel_tol, limit=200)
    return (m / (math.pi * math.sqrt(T))) * (1.0 - integral)


def bachelier_iv_log(m, Vbar, T, abs_tol=1e-12, rel_tol=1e-12):
    """sigma_N via the logarithmic finite-interval form (Eq. E.4)."""

    def imag_integrand(t):
        nu = nu_n(t)
        nup = nu_n_prime(t)
        V = bachelier_time_value(nu)
        ratio = (V - Vbar) / (V + Vbar)
        val = nup * cmath.log(ratio)
        return val.imag

    integral, _ = integrate.quad(imag_integrand, 0.0, 1.0,
                                  epsabs=abs_tol, epsrel=rel_tol, limit=200)
    return (m / (math.pi * math.sqrt(T))) * integral


# ---------------------------------------------------------------------------
# At-the-money closed form and sanity helpers
# ---------------------------------------------------------------------------

def bachelier_atm_iv(C, D, T):
    """At-the-money closed form sigma_N = sqrt(2*pi/T) * C / D.

    C is the (un-normalized) ATM call price, D a discount factor (D=1 if C
    is already forward/undiscounted); see module docstring for the
    derivation from C = sigma_N*sqrt(T)*phi(0).
    """
    return math.sqrt(2.0 * math.pi / T) * C / D


def bachelier_call_price(F, K, T, sigma_N):
    """Plain Bachelier call price C = (F-K)*Phi(d) + sigma_N*sqrt(T)*phi(d),
    d = (F-K)/(sigma_N*sqrt(T)); used only to manufacture test inputs."""
    denom = sigma_N * math.sqrt(T)
    d = (F - K) / denom
    return (F - K) * phi_cdf(d).real + denom * phi_pdf(d).real
