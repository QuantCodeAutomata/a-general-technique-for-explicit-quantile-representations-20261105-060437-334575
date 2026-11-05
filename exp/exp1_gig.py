"""exp_1: GIG quantile construction and the general contour method.

For every (a, b, sigma) in `src.config.GIG_TEST_PARAMS` and every probability
level in `GIG_PROB_LEVELS`, builds the p-independent GIG contour once (Steps
1-4, exp_1) and evaluates the quantile with both the logarithmic form (Eq. 1,
Step 5) and the residue form (Eq. 4), comparing each against a Newton-method
reference quantile (Appendix D.1). Writes `results/exp1_gig_quantiles.csv`
and `results/exp1_gig_errors.png`.
"""
from __future__ import annotations

import logging
import time

import mpmath as mp
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import config
from src import contour_core as cc
from src import gig

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp1_gig")


def run() -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for a, b, sigma in config.GIG_TEST_PARAMS:
        contour = gig.build_gig_contour(a, b, dps=config.CONTOUR_DPS)
        log.info("GIG contour built: a=%s b=%s, mesh nodes=%d, x_lo=%s, x_hi=%s",
                  a, b, len(contour.acc.mesh.xs),
                  mp.nstr(contour.acc.mesh.x_lo, 6), mp.nstr(contour.acc.mesh.x_hi, 6))
        for p in config.GIG_PROB_LEVELS:
            q_res = gig.gig_quantile(contour, p, dps=config.CONTOUR_DPS)
            q_log = gig.gig_quantile_log(contour, p, dps=config.CONTOUR_DPS)
            ref = gig.reference_quantile_gig(a, b, p, dps=config.NEWTON_REFERENCE_DPS)
            err_res = float(cc.relative_error(q_res, ref))
            err_log = float(cc.relative_error(q_log, ref))
            rows.append(dict(
                a=a, b=b, sigma=sigma, p=p,
                quantile_residue=float(sigma * q_res),
                quantile_log=float(sigma * q_log),
                quantile_reference=float(sigma * ref),
                rel_error_residue=err_res,
                rel_error_log=err_log,
            ))
    df = pd.DataFrame(rows)
    df.to_csv("results/exp1_gig_quantiles.csv", index=False)
    log.info("Wrote results/exp1_gig_quantiles.csv (%d rows) in %.1fs", len(df), time.time() - t0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (a, b), grp in df.groupby(["a", "b"]):
        ax.semilogy(grp["p"], grp["rel_error_residue"], marker="o", label=f"a={a}, b={b}")
    ax.set_xlabel("probability level p")
    ax.set_ylabel("relative error (residue form, Eq. 4)")
    ax.set_title("exp_1: GIG contour-integral quantile vs. Newton reference")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/exp1_gig_errors.png", dpi=150)
    plt.close(fig)

    max_err_res = df["rel_error_residue"].max()
    max_err_log = df["rel_error_log"].max()
    log.info("Max relative error: residue form %.3e, log form %.3e", max_err_res, max_err_log)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()
