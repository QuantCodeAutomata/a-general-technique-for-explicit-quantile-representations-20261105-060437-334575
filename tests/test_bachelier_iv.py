"""Tests for the Bachelier (normal) implied-vol contour integral (exp_9)."""
import math

import pytest
from scipy.optimize import brentq

from src import bachelier_iv as biv
from src import config


@pytest.mark.parametrize("m,Vbar,T", config.BACHELIER_TEST_CASES)
def test_bachelier_residue_log_agreement(m, Vbar, T):
    """Residue form (Eq. E.3) and logarithmic form (Eq. E.4) must agree to
    tight tolerance (both are double-precision scipy quadratures of the
    same root)."""
    sigma_residue = biv.bachelier_iv_residue(m, Vbar, T)
    sigma_log = biv.bachelier_iv_log(m, Vbar, T)
    assert abs(sigma_residue - sigma_log) < 1e-8


@pytest.mark.parametrize("m,Vbar,T", config.BACHELIER_TEST_CASES)
def test_bachelier_residue_matches_brentq_root(m, Vbar, T):
    """Cross-check against a plain scalar root-find of V(v) = Vbar (independent
    of the contour-integral method)."""
    v_root = brentq(lambda v: biv.bachelier_time_value(v).real - Vbar,
                     1e-8, 100.0, xtol=1e-13, rtol=1e-13)
    sigma_brentq = v_root * m / math.sqrt(T)
    sigma_residue = biv.bachelier_iv_residue(m, Vbar, T)
    assert abs(sigma_residue - sigma_brentq) < 1e-8


def test_bachelier_time_value_gammainc_cross_check():
    """V(v) = v*phi(1/v) - Phi(-1/v) must equal
    Gamma(-1/2, 1/(2v^2))/(4*sqrt(pi)) (module docstring, both forms)."""
    for v in [0.3, 1.0, 2.5, 5.0]:
        direct = biv.bachelier_time_value(v).real
        via_gammainc = float(biv.bachelier_time_value_gammainc(v))
        assert abs(direct - via_gammainc) < 1e-10


def test_bachelier_atm_closed_form_consistency():
    """bachelier_atm_iv(C, D, T) must invert the ATM Bachelier price formula
    C = sigma_N*sqrt(T)*phi(0)/D exactly it was derived from (manufactured
    C from a known sigma_true; see src/bachelier_iv.py module docstring)."""
    for sigma_true, T in [(0.2, 1.0), (0.5, 2.0), (0.1, 0.25), (1.0, 3.0)]:
        D = 1.0
        C = sigma_true * math.sqrt(T) / math.sqrt(2.0 * math.pi) * D
        sigma_recovered = biv.bachelier_atm_iv(C, D, T)
        assert abs(sigma_recovered - sigma_true) < 1e-12


def test_bachelier_atm_matches_general_bachelier_call_price():
    """bachelier_call_price at F=K (d=0) must equal the ATM closed form's
    assumed C = sigma_N*sqrt(T)*phi(0)."""
    sigma_true, T = 0.3, 1.5
    F = K = 100.0
    C = biv.bachelier_call_price(F, K, T, sigma_true)
    sigma_recovered = biv.bachelier_atm_iv(C, 1.0, T)
    assert abs(sigma_recovered - sigma_true) < 1e-10


def test_bachelier_continuity_sigma_to_zero_as_vbar_to_zero():
    """Zero time value gives zero implied volatility by continuity
    (methodology, exp_9): sigma_N must decrease monotonically and -> 0 as
    Vbar -> 0+."""
    m, T = 1.0, 1.0
    vbars = [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-20, 1e-40]
    sigmas = [biv.bachelier_iv_residue(m, Vbar, T) for Vbar in vbars]
    assert all(s2 < s1 for s1, s2 in zip(sigmas, sigmas[1:]))
    assert sigmas[-1] < 0.1
    assert sigmas[-1] > 0.0
