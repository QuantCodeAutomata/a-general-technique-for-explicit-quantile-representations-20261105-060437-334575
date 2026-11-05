"""exp_9: Bachelier (normal) implied volatility via the contour integral
(Schadner 2026, exp_9).

For every (m, Vbar, T) in `config.BACHELIER_TEST_CASES`, recovers sigma_N with
both the residue form (Eq. E.3) and the logarithmic finite-interval form
(Eq. E.4) and checks they agree. Also runs two sanity checks from the
methodology: (1) the at-the-money closed form against a manufactured ATM
price, and (2) continuity of sigma_N -> 0 as Vbar -> 0+.

Writes `results/exp9_bachelier_iv.csv` and
`results/exp9_bachelier_sanity_checks.csv`.
"""
from __future__ import annotations

import logging
import math
import os
import time

import pandas as pd
from scipy.optimize import brentq

from src import bachelier_iv as biv
from src import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp9_bachelier_iv")


def run() -> pd.DataFrame:
    t0 = time.time()
    rows = []
    for m, Vbar, T in config.BACHELIER_TEST_CASES:
        sigma_residue = biv.bachelier_iv_residue(m, Vbar, T)
        sigma_log = biv.bachelier_iv_log(m, Vbar, T)
        agreement_abs_diff = abs(sigma_residue - sigma_log)
        rows.append(dict(m=m, Vbar=Vbar, T=T,
                          sigma_N_residue=sigma_residue, sigma_N_log=sigma_log,
                          agreement_abs_diff=agreement_abs_diff))
        log.info("m=%.3g Vbar=%.3g T=%.3g: sigma_N residue=%.12g log=%.12g diff=%.3e",
                  m, Vbar, T, sigma_residue, sigma_log, agreement_abs_diff)

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/exp9_bachelier_iv.csv", index=False)
    log.info("Wrote results/exp9_bachelier_iv.csv (%d rows) in %.1fs", len(df), time.time() - t0)
    log.info("Max residue-vs-log agreement abs diff: %.3e", df["agreement_abs_diff"].max())

    sanity_rows = []

    # --- At-the-money closed-form check ---
    # bachelier_atm_iv(C, D, T) = sqrt(2*pi/T) * C/D is tested in isolation
    # against its own derivation: manufacture an ATM call price C from a
    # known sigma_true via the plain Bachelier formula at F=K (d=0), then
    # recover sigma_true from C through bachelier_atm_iv (see module
    # docstrings in src/bachelier_iv.py; the m -> 0 degenerate limit of the
    # general (m, Vbar, T) solver is not used directly since v_N = sigma*sqrt(T)/m
    # diverges as m -> 0).
    for sigma_true, T in [(0.2, 1.0), (0.5, 2.0), (0.1, 0.25)]:
        D = 1.0
        C = sigma_true * math.sqrt(T) / math.sqrt(2.0 * math.pi) * D  # F=K ATM price, d=0
        sigma_atm = biv.bachelier_atm_iv(C, D, T)
        atm_abs_diff = abs(sigma_atm - sigma_true)
        sanity_rows.append(dict(check="atm_closed_form", sigma_true=sigma_true, T=T,
                                 C=C, D=D, sigma_recovered=sigma_atm, abs_diff=atm_abs_diff))
        log.info("ATM check: sigma_true=%.4g T=%.4g -> sigma_atm=%.12g (abs diff %.3e)",
                  sigma_true, T, sigma_atm, atm_abs_diff)

    # Independent cross-check of the general (m, Vbar, T) solver against a
    # plain scalar brentq root of V(v) = Vbar for a manufactured, non-degenerate
    # (m, Vbar, T) triple built from a known sigma_true.
    sigma_true, m, T = 0.3, 1.0, 1.0
    v_true = sigma_true * math.sqrt(T) / m
    Vbar_manufactured = biv.bachelier_time_value(v_true).real
    v_brentq = brentq(lambda v: biv.bachelier_time_value(v).real - Vbar_manufactured,
                       1e-8, 50.0, xtol=1e-14, rtol=1e-14)
    sigma_brentq = v_brentq * m / math.sqrt(T)
    sigma_residue_check = biv.bachelier_iv_residue(m, Vbar_manufactured, T)
    brentq_abs_diff = abs(sigma_residue_check - sigma_brentq)
    sanity_rows.append(dict(check="general_solver_vs_brentq", sigma_true=sigma_true, T=T,
                             m=m, Vbar=Vbar_manufactured, sigma_recovered=sigma_residue_check,
                             abs_diff=brentq_abs_diff))
    log.info("General-solver-vs-brentq check: sigma_true=%.4g -> residue=%.12g brentq=%.12g "
              "(abs diff %.3e)", sigma_true, sigma_residue_check, sigma_brentq, brentq_abs_diff)

    # --- Continuity check: Vbar -> 0+ should give sigma_N -> 0 ---
    m, T = 1.0, 1.0
    prev_sigma = None
    for Vbar in [1e-1, 1e-3, 1e-6, 1e-9, 1e-12, 1e-20]:
        sigma_N = biv.bachelier_iv_residue(m, Vbar, T)
        monotone_decreasing = (prev_sigma is None) or (sigma_N < prev_sigma)
        sanity_rows.append(dict(check="continuity", Vbar=Vbar, sigma_N=sigma_N,
                                 monotone_decreasing=monotone_decreasing))
        log.info("Continuity check: Vbar=%.3e -> sigma_N=%.6g (monotone decreasing: %s)",
                  Vbar, sigma_N, monotone_decreasing)
        prev_sigma = sigma_N

    df_sanity = pd.DataFrame(sanity_rows)
    df_sanity.to_csv("results/exp9_bachelier_sanity_checks.csv", index=False)
    log.info("Wrote results/exp9_bachelier_sanity_checks.csv (%d rows)", len(df_sanity))

    return df


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run()
