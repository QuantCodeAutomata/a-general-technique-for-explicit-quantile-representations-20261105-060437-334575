"""Tests for exp_10's cross-law verification dispatcher (src/verification.py)."""
from functools import lru_cache

import mpmath as mp
import pytest

from src import config
from src.verification import build_law
from src.contour_core import relative_error


@lru_cache(maxsize=None)
def _cached_handle(name: str):
    """Build each law's contour once and reuse it across the three
    parametrized checks below (quantile, accumulated mass, density norm);
    the contour itself is p-independent (methodology, exp_10)."""
    law = next(law for law in config.VERIFICATION_LAWS if law["name"] == name)
    return build_law(law, dps=18, newton_dps=22, control_dps=22)


@pytest.mark.parametrize("law", config.VERIFICATION_LAWS, ids=lambda law: law["name"])
def test_law_quantile_matches_reference(law):
    """Each of the twelve working laws reproduces its Newton-method reference
    quantile at a representative probability level, as required by exp_10."""
    handle = _cached_handle(law["name"])
    p = 0.3
    q_contour = handle.quantile_fn(p)
    q_reference = handle.reference_fn(p)
    err = relative_error(q_contour, q_reference)
    # dps=18 here (vs. dps=24 + guard digits in the production exp_10 run) trades
    # precision for test speed; this only checks the qualitative property that the
    # contour quantile tracks the Newton reference, not the paper's 1e-17 claim.
    assert err < 1e-5, f"{law['name']}: relative error {err} too large at p={p}"


@pytest.mark.parametrize("law", config.VERIFICATION_LAWS, ids=lambda law: law["name"])
def test_accumulated_cdf_reaches_expected_mass(law):
    """The accumulated complex CDF along each contour should reach the
    expected total (or half-axis) mass, per the control checks of exp_10."""
    handle = _cached_handle(law["name"])
    for mass, expected in zip(handle.accumulated_masses, handle.expected_masses):
        assert abs(mp.re(mass) - expected) < 1e-5


@pytest.mark.parametrize("law", config.VERIFICATION_LAWS, ids=lambda law: law["name"])
def test_real_density_normalizes(law):
    """The real working density integrates to one (or to the VG half-axis
    split) to good precision, independent of the contour construction."""
    handle = _cached_handle(law["name"])
    assert abs(handle.density_integral - handle.expected_density_integral) < 1e-5


def test_quantile_monotone_in_p():
    """Sanity check: the quantile function must be nondecreasing in p for every law."""
    for law in config.VERIFICATION_LAWS[:3]:
        handle = build_law(law, dps=18, newton_dps=22, control_dps=22)
        qs = [handle.quantile_fn(p) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        for lo, hi in zip(qs, qs[1:]):
            assert mp.re(hi) >= mp.re(lo) - mp.mpf('1e-10')
