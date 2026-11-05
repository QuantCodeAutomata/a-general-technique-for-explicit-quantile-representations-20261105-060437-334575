"""exp_5: Normal and log-normal quantile construction.

For every (mu, sigma) in `src.config.NORMAL_TEST_PARAMS` and every
probability level in `NORMAL_PROB_LEVELS`, builds the p-independent
standard-normal working contour once (methodology, exp_5) and reconstructs
the Normal quantile Q~(p) = mu + sigma*Q(p) via the affine map, comparing it
against a SciPy-seeded Newton-method reference. Separately, builds the
standard-normal working contour ONCE more and reuses it across every
(sigma, a) in `LOGNORMAL_TEST_PARAMS` x `NORMAL_PROB_LEVELS` for the
log-normal reconstruction Q~(p) = sigma*exp(a*Q(p)) via the exponential map
-- the whole point of this experiment being that log-normal quantiles come
from the identical p-independent normal contour via a different
reconstruction map, not a separately built one.

Writes `results/exp5_normal_quantiles.csv`, `results/exp5_lognormal_quantiles.csv`
and `results/exp5_normal_lognormal_errors.png`.
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
from src import normal_lognormal as nl

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp5_normal_lognormal")


def run() -> pd.DataFrame:
    t0 = time.time()

    # --- Normal ---
    normal_rows = []
    for mu, sigma in config.NORMAL_TEST_PARAMS:
        contour = nl.build_normal_contour(dps=config.CONTOUR_DPS)
        log.info("Normal contour built: mu=%s sigma=%s, mesh nodes=%d, x_lo=%s, x_hi=%s",
                  mu, sigma, len(contour.acc.mesh.xs),
                  mp.nstr(contour.acc.mesh.x_lo, 6), mp.nstr(contour.acc.mesh.x_hi, 6))
        for p in config.NORMAL_PROB_LEVELS:
            q = nl.normal_quantile(contour, p, mu=mu, sigma=sigma, dps=config.CONTOUR_DPS)
            ref = nl.reference_quantile_normal(mu, sigma, p, dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q, ref))
            normal_rows.append(dict(
                mu=mu, sigma=sigma, p=p,
                quantile_contour=float(q),
                quantile_reference=float(ref),
                rel_error=err,
            ))
    normal_df = pd.DataFrame(normal_rows)
    normal_df.to_csv("results/exp5_normal_quantiles.csv", index=False)
    log.info("Wrote results/exp5_normal_quantiles.csv (%d rows)", len(normal_df))

    # --- Log-normal (reuses a single standard-normal contour) ---
    lognormal_rows = []
    std_contour = nl.build_normal_contour(dps=config.CONTOUR_DPS)
    log.info("Standard-normal contour built (reused for log-normal): mesh nodes=%d, x_lo=%s, x_hi=%s",
              len(std_contour.acc.mesh.xs),
              mp.nstr(std_contour.acc.mesh.x_lo, 6), mp.nstr(std_contour.acc.mesh.x_hi, 6))
    for sigma, a in config.LOGNORMAL_TEST_PARAMS:
        for p in config.NORMAL_PROB_LEVELS:
            q = nl.lognormal_quantile(std_contour, p, sigma=sigma, a=a, dps=config.CONTOUR_DPS)
            ref = nl.reference_quantile_lognormal(sigma, a, p, dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q, ref))
            lognormal_rows.append(dict(
                sigma=sigma, a=a, p=p,
                quantile_contour=float(q),
                quantile_reference=float(ref),
                rel_error=err,
            ))
    lognormal_df = pd.DataFrame(lognormal_rows)
    lognormal_df.to_csv("results/exp5_lognormal_quantiles.csv", index=False)
    log.info("Wrote results/exp5_lognormal_quantiles.csv (%d rows) in %.1fs",
              len(lognormal_df), time.time() - t0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (mu, sigma), grp in normal_df.groupby(["mu", "sigma"]):
        axes[0].semilogy(grp["p"], grp["rel_error"], marker="o", label=f"mu={mu}, sigma={sigma}")
    axes[0].set_xlabel("probability level p")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("Normal quantile vs. SciPy-seeded Newton reference")
    axes[0].legend(fontsize=7)

    for (sigma, a), grp in lognormal_df.groupby(["sigma", "a"]):
        axes[1].semilogy(grp["p"], grp["rel_error"], marker="o", label=f"sigma={sigma}, a={a}")
    axes[1].set_xlabel("probability level p")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("Log-normal quantile vs. SciPy-seeded Newton reference")
    axes[1].legend(fontsize=7)

    fig.suptitle("exp_5: Normal and log-normal contour-integral quantile errors")
    fig.tight_layout()
    fig.savefig("results/exp5_normal_lognormal_errors.png", dpi=150)
    plt.close(fig)

    max_err_normal = normal_df["rel_error"].max()
    max_err_lognormal = lognormal_df["rel_error"].max()
    log.info("Max relative error: normal %.3e, log-normal %.3e", max_err_normal, max_err_lognormal)
    return normal_df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()
