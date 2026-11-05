"""Cross-law verification of the contour-integral quantile method (exp_10).

Dispatches the twelve working laws of ``config.VERIFICATION_LAWS`` to the
per-family contour implementations already built for exp_1-exp_7, and adds
the control checks described in the methodology of exp_10: the accumulated
complex CDF along each contour should reach its total mass to about 1e-24,
and the real working density should integrate to one (or, for the Variance
Gamma half-axis split, to the boundary mass p0 / 1-p0) to about 1e-30.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mpmath as mp

from src import config
from src import gig, gb, gg_gev, gh
from src import normal_lognormal as nl
from src import student_cauchy as sc
from src import vg


@dataclass
class LawHandle:
    """A built, p-independent contour plus the callables needed to verify it."""
    name: str
    quantile_fn: Callable[[float], mp.mpf]       # p -> contour-integral quantile
    reference_fn: Callable[[float], mp.mpf]      # p -> Newton-method reference quantile
    accumulated_masses: list                      # accumulated CDF(s) at right truncation
    expected_masses: list                          # the mass(es) each should approach
    density_integral: mp.mpf                       # control check: real density integrates to 1 (or p0/1-p0)
    expected_density_integral: mp.mpf


def build_law(law: dict, dps: int = config.CONTOUR_DPS,
              newton_dps: int = config.NEWTON_REFERENCE_DPS,
              control_dps: int = config.CONTROL_CHECK_DPS) -> LawHandle:
    """Build the p-independent contour for one working law and its reference.

    ``law`` is one entry of ``config.VERIFICATION_LAWS``: ``{"name", "family", "params"}``.
    """
    family, params, name = law["family"], law["params"], law["name"]

    if family == "beta":
        a, b = params["a"], params["b"]
        contour = gb.build_beta_contour(a, b, dps=dps)
        quantile_fn = lambda p: gb.beta_quantile(contour, p, dps=dps)
        reference_fn = lambda p: gb.reference_quantile_beta(a, b, p, dps=newton_dps)
        masses = [contour.acc.P_vals[-1]]
        expected = [mp.mpf(1)]
        with mp.workdps(control_dps):
            density_integral = mp.quad(lambda x: gb.g_beta(mp.mpf(x), a, b).real, [0, 1])
        expected_density = mp.mpf(1)

    elif family == "gamma":
        a = params["a"]
        contour = gg_gev.build_gamma_contour(a, dps=dps)
        quantile_fn = lambda p: gg_gev.gamma_quantile(contour, p, dps=dps)
        # Plain unit-rate gamma working law == GG reconstruction at c=1, sigma=1.
        reference_fn = lambda p: gg_gev.reference_quantile_gg(a, 1.0, p, sigma=1.0, dps=newton_dps)
        masses = [contour.acc.P_vals[-1]]
        expected = [mp.mpf(1)]
        with mp.workdps(control_dps):
            density_integral = mp.quad(lambda x: gg_gev.g_gamma(mp.mpf(x), a).real, [0, mp.inf])
        expected_density = mp.mpf(1)

    elif family == "gig":
        a, b = params["a"], params["b"]
        contour = gig.build_gig_contour(a, b, dps=dps)
        quantile_fn = lambda p: gig.gig_quantile(contour, p, dps=dps)
        reference_fn = lambda p: gig.reference_quantile_gig(a, b, p, dps=newton_dps)
        masses = [contour.acc.P_vals[-1]]
        expected = [mp.mpf(1)]
        with mp.workdps(control_dps):
            density_integral = mp.quad(lambda x: gig.g_gig(mp.mpf(x), a, b).real, [0, mp.inf])
        expected_density = mp.mpf(1)

    elif family == "gh":
        a, b, d = params["a"], params["b"], params["d"]
        contour = gh.build_gh_contour(a, b, d, dps=dps)
        quantile_fn = lambda p: gh.gh_quantile(contour, p, dps=dps)
        reference_fn = lambda p: gh.reference_quantile_gh(a, b, d, p, dps=newton_dps)
        masses = [contour.acc.P_vals[-1]]
        expected = [mp.mpf(1)]
        with mp.workdps(control_dps):
            density_integral = mp.quad(lambda x: gh.g_gh(mp.mpf(x), a, b, d).real, [-mp.inf, mp.inf])
        expected_density = mp.mpf(1)

    elif family == "normal":
        contour = nl.build_normal_contour(dps=dps)
        quantile_fn = lambda p: nl.normal_standard_quantile(contour, p, dps=dps)
        reference_fn = lambda p: nl.reference_quantile_normal(0.0, 1.0, p, dps=newton_dps)
        masses = [contour.acc.P_vals[-1]]
        expected = [mp.mpf(1)]
        with mp.workdps(control_dps):
            density_integral = mp.quad(lambda x: nl.g_normal(mp.mpf(x)).real, [-mp.inf, mp.inf])
        expected_density = mp.mpf(1)

    elif family == "student_t":
        a = params["a"]
        contour = sc.build_student_t_contour(a, dps=dps)
        quantile_fn = lambda p: sc.student_t_quantile(contour, p, dps=dps)
        reference_fn = lambda p: sc.reference_quantile_student_t(a, p, dps=newton_dps)
        masses = [contour.acc.P_vals[-1]]
        expected = [mp.mpf(1)]
        with mp.workdps(control_dps):
            density_integral = mp.quad(lambda x: sc.g_student_t(mp.mpf(x), a).real, [-mp.inf, mp.inf])
        expected_density = mp.mpf(1)

    elif family == "vg":
        a, d = params["a"], params["d"]
        contour = vg.build_vg_contour(a, d, dps=dps)
        quantile_fn = lambda p: vg.vg_quantile(contour, p, dps=dps)
        reference_fn = lambda p: vg.reference_quantile_vg(a, d, p, dps=newton_dps)
        masses = [contour.acc_plus.P_vals[-1], contour.acc_minus.P_vals[-1]]
        expected = [mp.mpf(1) - contour.p0, contour.p0]
        with mp.workdps(control_dps):
            density_integral = (
                mp.quad(lambda x: vg.g_vg(mp.mpf(x), a, d, 1).real, [mp.mpf('1e-14'), mp.inf])
                + mp.quad(lambda x: vg.g_vg(mp.mpf(x), a, d, -1).real, [mp.mpf('1e-14'), mp.inf])
            )
        expected_density = mp.mpf(1)

    else:
        raise ValueError(f"unknown working-law family: {family}")

    return LawHandle(name, quantile_fn, reference_fn, masses, expected,
                      density_integral, expected_density)
