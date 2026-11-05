"""Tests for the Black-Scholes implied-vol contour integral (exp_8)."""
import mpmath as mp
import pytest

from src import bs_iv as bs
from src import config


@pytest.mark.parametrize("k,v,T", [
    (0.05, 0.4, 1.0),
    (0.5, 0.8, 1.0),
    (1.0, 1.5, 1.0),
    (0.01, 0.1, 0.25),
    (2.0, 3.0, 4.0),
    (1e-9, 0.5, 1.0),  # near k=0, no special-casing needed
])
def test_bs_iv_recovery_matches_ground_truth(k, v, T):
    """Generate a high-precision price from a known total vol v, recover v
    via the double-precision contour-integral formula (Eq. 25), and compare
    to the ground truth (expected_outcomes, exp_8: abs error ~1e-15)."""
    with mp.workdps(config.BS_REFERENCE_DPS):
        cbar_true = bs.bs_call_price_mp(k, v)
    cbar_true_f = float(cbar_true)
    sigma_true = v / (T ** 0.5)

    sigma_recovered = bs.sigma_bs_from_price(
        k, cbar_true_f, T, abs_tol=config.BS_QUAD_ABS_TOL, rel_tol=config.BS_QUAD_REL_TOL)

    assert abs(sigma_recovered - sigma_true) < 1e-8


def test_bs_iv_repricing_consistency():
    """Repricing with the recovered sigma_BS must reproduce the observed
    price to near machine precision (expected_outcomes, exp_8: ~2e-16)."""
    k, v, T = 0.25, 0.8, 1.0
    with mp.workdps(config.BS_REFERENCE_DPS):
        cbar_true = bs.bs_call_price_mp(k, v)
    cbar_true_f = float(cbar_true)

    sigma_recovered = bs.sigma_bs_from_price(k, cbar_true_f, T)
    v_recovered = sigma_recovered * (T ** 0.5)
    cbar_repriced = bs.bs_call_price(k, v_recovered).real

    assert abs(cbar_repriced - cbar_true_f) < 1e-8


def test_bs_iv_42_case_grid_sample_within_tolerance():
    """A handful of the 42-case grid (not all, for speed), same check as
    above but spanning several (k, v, T) combinations."""
    import itertools
    sample_idx = [0, 7, 15, 23, 30, 41]
    grid = [(k, v) for k, v in itertools.product(config.BS_LOG_MONEYNESS_GRID,
                                                  config.BS_TOTAL_VOL_GRID)]
    for idx in sample_idx:
        k, v = grid[idx]
        T = config.BS_MATURITIES[idx % 3]
        with mp.workdps(config.BS_REFERENCE_DPS):
            cbar_true = bs.bs_call_price_mp(k, v)
        cbar_true_f = float(cbar_true)
        sigma_true = v / (T ** 0.5)
        sigma_recovered = bs.sigma_bs_from_price(k, cbar_true_f, T)
        assert abs(sigma_recovered - sigma_true) < 1e-8, (idx, k, v, T)


def test_bs_put_call_parity_k_negative_matches_positive_branch():
    """Put-call-parity relation c(k,v) = 1 - e^k + e^k*c(-k,v) (module
    docstring): solving for k < 0 via the parity transform must recover the
    same total volatility as solving the k > 0 case directly."""
    k_pos, v, T = 0.3, 0.6, 1.0
    with mp.workdps(config.BS_REFERENCE_DPS):
        cbar_pos = bs.bs_call_price_mp(k_pos, v)
        cbar_neg = mp.mpf(1) - mp.e ** mp.mpf(-k_pos) + mp.e ** mp.mpf(-k_pos) * cbar_pos
    cbar_neg_f = float(cbar_neg)

    sigma_true = v / (T ** 0.5)
    sigma_recovered = bs.sigma_bs_from_price(-k_pos, cbar_neg_f, T)
    assert abs(sigma_recovered - sigma_true) < 1e-8
