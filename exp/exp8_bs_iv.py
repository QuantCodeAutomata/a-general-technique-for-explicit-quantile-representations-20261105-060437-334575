"""exp_8: Black-Scholes implied volatility via the finite-interval contour
integral (Schadner 2026, exp_8).

For every (k, v, T) in the 42-case grid built from `config.BS_LOG_MONEYNESS_GRID`
x `config.BS_TOTAL_VOL_GRID` (maturities assigned cyclically from
`config.BS_MATURITIES`), computes the TRUE normalized call price at
dps=`config.BS_REFERENCE_DPS` (mpmath), recovers the implied total volatility
with the double-precision contour-integral formula (Eq. 25, `src.bs_iv`), and
compares the recovered volatility and a repriced call to the ground truth.
Also runs an independent mpmath cross-check (Eq. 25, evaluated directly with
`mp.quad` instead of `scipy.integrate.quad`) at dps=35 and dps=50 for the
paper's three example cases.

Writes `results/exp8_bs_iv_grid.csv`, `results/exp8_bs_iv_independent_check.csv`
and `results/exp8_bs_iv_errors.png`.
"""
from __future__ import annotations

import itertools
import logging
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import pandas as pd

from src import bs_iv
from src import config

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp8_bs_iv")


def _build_grid():
    """42 (k, v, T) triples (methodology, exp_8): 7 log-moneyness x 6 total-vol
    combinations, maturity assigned cyclically from BS_MATURITIES by case index."""
    grid = []
    for idx, (k, v) in enumerate(itertools.product(config.BS_LOG_MONEYNESS_GRID,
                                                     config.BS_TOTAL_VOL_GRID)):
        T = config.BS_MATURITIES[idx % 3]
        grid.append((k, v, T))
    return grid


def _grid_case(k, v, T):
    with mp.workdps(config.BS_REFERENCE_DPS):
        cbar_true_mp = bs_iv.bs_call_price_mp(k, v)
    cbar_true_f = float(cbar_true_mp)
    sigma_true = v / (T ** 0.5)

    sigma_recovered = bs_iv.sigma_bs_from_price(
        k, cbar_true_f, T, abs_tol=config.BS_QUAD_ABS_TOL, rel_tol=config.BS_QUAD_REL_TOL)

    abs_error = abs(sigma_recovered - sigma_true)
    rel_error = abs_error / sigma_true if sigma_true != 0 else abs_error

    v_recovered = sigma_recovered * (T ** 0.5)
    cbar_repriced = bs_iv.bs_call_price(k, v_recovered).real
    reprice_abs_error = abs(cbar_repriced - cbar_true_f)

    return dict(k=k, v=v, T=T, sigma_true=sigma_true, sigma_recovered=sigma_recovered,
                abs_error=abs_error, rel_error=rel_error, reprice_abs_error=reprice_abs_error)


def _independent_check(k, v):
    """mpmath-only cross-check of Eq. 25 at dps=35 and dps=50 against the
    dps=100 ground-truth volatility, for the paper's three example cases."""
    with mp.workdps(config.BS_REFERENCE_DPS):
        cbar_true_mp = bs_iv.bs_call_price_mp(k, v)
        sigma_true = mp.mpf(v) / mp.sqrt(mp.mpf(1.0))  # T=1 for these example cases (see run())

    rows = []
    for dps in config.BS_INDEPENDENT_CHECK_DPS:
        with mp.workdps(dps):
            cbar_dps = +cbar_true_mp  # re-round to current precision
            sigma_mp = bs_iv.sigma_bs_from_price_mp(k, cbar_dps, 1.0, dps=dps)
        rel_discrepancy = abs((sigma_mp - sigma_true) / sigma_true)
        rows.append(dict(k=k, v=v, dps=dps,
                          sigma_true=float(sigma_true), sigma_mpmath=float(sigma_mp),
                          rel_discrepancy=float(rel_discrepancy)))
    return rows


def run() -> pd.DataFrame:
    t0 = time.time()
    grid = _build_grid()
    log.info("Running exp_8 Black-Scholes IV recovery over %d grid cases", len(grid))

    rows = [_grid_case(k, v, T) for k, v, T in grid]
    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/exp8_bs_iv_grid.csv", index=False)
    log.info("Wrote results/exp8_bs_iv_grid.csv (%d rows) in %.1fs", len(df), time.time() - t0)
    log.info("Max abs error: %.3e, max rel error: %.3e, max reprice abs error: %.3e",
              df["abs_error"].max(), df["rel_error"].max(), df["reprice_abs_error"].max())

    # Independent mpmath dps=35/50 cross-check, paper's three example cases
    # (fixed at T=1, as specified by the methodology for these examples).
    example_cases = [(0.05, 0.4), (0.5, 0.8), (1.0, 1.5)]
    indep_rows = []
    for k, v in example_cases:
        indep_rows.extend(_independent_check(k, v))
    df_indep = pd.DataFrame(indep_rows)
    df_indep.to_csv("results/exp8_bs_iv_independent_check.csv", index=False)
    log.info("Wrote results/exp8_bs_iv_independent_check.csv (%d rows)", len(df_indep))
    log.info("Independent check max rel discrepancy: %.3e", df_indep["rel_discrepancy"].max())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(range(len(df)), df["abs_error"], marker="o", ms=3, label="abs error (sigma_BS)")
    ax.semilogy(range(len(df)), df["reprice_abs_error"], marker="x", ms=3, label="reprice abs error")
    ax.set_xlabel("grid case index")
    ax.set_ylabel("absolute error")
    ax.set_title("exp_8: Black-Scholes implied-vol contour-integral recovery errors")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/exp8_bs_iv_errors.png", dpi=150)
    plt.close(fig)

    return df


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run()
