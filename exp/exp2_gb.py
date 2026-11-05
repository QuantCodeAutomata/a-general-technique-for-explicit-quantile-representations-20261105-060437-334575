"""exp_2: Generalized Beta (GB) quantile construction and the general
contour method.

For every (a, b) in `config.GB_BETA_TEST_PARAMS`, builds the p-independent
Beta working contour once and evaluates the quantile at every probability
level in `config.GB_PROB_LEVELS`, comparing against a Newton-method
reference (methodology, exp_2). Separately, for every (a, b, c, rho, sigma)
in `config.GB_FAMILY_TEST_PARAMS`, applies the reconstruction map on top of
the same kind of Beta working contour and compares against the reference
transformed by the same map. Writes `results/exp2_gb_quantiles.csv` and
`results/exp2_gb_errors.png`.
"""
from __future__ import annotations

import logging
import os
import time

import mpmath as mp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src import contour_core as cc
from src import gb

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp2_gb")


def run() -> pd.DataFrame:
    rows = []
    t0 = time.time()

    # Part 1: pure Beta working law (sigma=1, c=1, rho=0 reconstruction).
    for a, b in config.GB_BETA_TEST_PARAMS:
        contour = gb.build_beta_contour(a, b, dps=config.CONTOUR_DPS)
        log.info("Beta contour built: a=%s b=%s, mesh nodes=%d, x_lo=%s, x_hi=%s",
                  a, b, len(contour.acc.mesh.xs),
                  mp.nstr(contour.acc.mesh.x_lo, 6), mp.nstr(contour.acc.mesh.x_hi, 6))
        for p in config.GB_PROB_LEVELS:
            q = gb.beta_quantile(contour, p, dps=config.CONTOUR_DPS)
            ref = gb.reference_quantile_beta(a, b, p, dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q, ref))
            rows.append(dict(
                group="beta", a=a, b=b, c=1.0, rho=0.0, sigma=1.0, p=p,
                quantile_contour=float(q), quantile_reference=float(ref), rel_error=err,
            ))

    # Part 2: full GB family reconstruction map.
    for a, b, c, rho, sigma in config.GB_FAMILY_TEST_PARAMS:
        contour = gb.build_beta_contour(a, b, dps=config.CONTOUR_DPS)
        log.info("GB-family Beta contour built: a=%s b=%s c=%s rho=%s sigma=%s, mesh nodes=%d",
                  a, b, c, rho, sigma, len(contour.acc.mesh.xs))
        for p in config.GB_PROB_LEVELS:
            q = gb.gb_quantile(contour, p, c=c, rho=rho, sigma=sigma, dps=config.CONTOUR_DPS)
            ref = gb.reference_quantile_gb(a, b, c, rho, sigma, p, dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q, ref))
            rows.append(dict(
                group="gb_family", a=a, b=b, c=c, rho=rho, sigma=sigma, p=p,
                quantile_contour=float(q), quantile_reference=float(ref), rel_error=err,
            ))

    df = pd.DataFrame(rows)
    df.to_csv("results/exp2_gb_quantiles.csv", index=False)
    log.info("Wrote results/exp2_gb_quantiles.csv (%d rows) in %.1fs", len(df), time.time() - t0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (group, a, b, c, rho, sigma), grp in df.groupby(["group", "a", "b", "c", "rho", "sigma"]):
        label = f"{group}: a={a},b={b},c={c},rho={rho},sigma={sigma}"
        ax.semilogy(grp["p"], grp["rel_error"], marker="o", label=label)
    ax.set_xlabel("probability level p")
    ax.set_ylabel("relative error")
    ax.set_title("exp_2: GB contour-integral quantile vs. Newton reference")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig("results/exp2_gb_errors.png", dpi=150)
    plt.close(fig)

    max_err = df["rel_error"].max()
    log.info("Max relative error (all groups): %.3e", max_err)
    return df


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run()
