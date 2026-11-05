"""exp_3: Generalized Gamma (GG) and Generalized Extreme Value (GEV) quantile
construction.

For every (a, c) in `src.config.GG_TEST_PARAMS` and every probability level
in `GG_GEV_PROB_LEVELS`, builds the p-independent unit-rate gamma(a) working
contour once (methodology, exp_3) and reconstructs the GG quantile
Q~_GG(p) = sigma * [Q(p)]^(1/c), comparing it against a gengamma-seeded
Newton-method reference. Separately, for every (a, sigma, mu) in
`GEV_TEST_PARAMS`, builds the gamma(a=1) [exponential] working contour once
and reconstructs the GEV quantile via the decreasing map applied to
Q(1-p), comparing it against the elementary closed form the paper's own
reconstruction reduces to (trusted as ground truth for GEV; SciPy's
`genextreme` is only an informational sanity check, included as an extra
column since its shape convention is opposite-signed, c = -a).

Writes `results/exp3_gg_quantiles.csv`, `results/exp3_gev_quantiles.csv` and
`results/exp3_gg_gev_errors.png`.
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
from src import gg_gev

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp3_gg_gev")


def run() -> pd.DataFrame:
    t0 = time.time()

    # --- Generalized Gamma (GG) ---
    gg_rows = []
    gg_contours = {}
    for a, c in config.GG_TEST_PARAMS:
        if a not in gg_contours:
            contour = gg_gev.build_gamma_contour(a, dps=config.CONTOUR_DPS)
            gg_contours[a] = contour
            log.info("gamma(a=%s) contour built: mesh nodes=%d, x_lo=%s, x_hi=%s",
                      a, len(contour.acc.mesh.xs),
                      mp.nstr(contour.acc.mesh.x_lo, 6), mp.nstr(contour.acc.mesh.x_hi, 6))
        contour = gg_contours[a]
        for p in config.GG_GEV_PROB_LEVELS:
            q = gg_gev.gg_quantile(contour, p, c=c, sigma=gg_gev.GG_SIGMA, dps=config.CONTOUR_DPS)
            ref = gg_gev.reference_quantile_gg(a, c, p, sigma=gg_gev.GG_SIGMA,
                                                dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q, ref))
            gg_rows.append(dict(
                a=a, c=c, sigma=gg_gev.GG_SIGMA, p=p,
                quantile_contour=float(q),
                quantile_reference=float(ref),
                rel_error=err,
            ))
    gg_df = pd.DataFrame(gg_rows)
    gg_df.to_csv("results/exp3_gg_quantiles.csv", index=False)
    log.info("Wrote results/exp3_gg_quantiles.csv (%d rows)", len(gg_df))

    # --- Generalized Extreme Value (GEV) ---
    gev_rows = []
    exp_contour = gg_gev.build_gamma_contour(1.0, dps=config.CONTOUR_DPS)
    log.info("gamma(a=1)[exponential] contour built: mesh nodes=%d, x_lo=%s, x_hi=%s",
              len(exp_contour.acc.mesh.xs),
              mp.nstr(exp_contour.acc.mesh.x_lo, 6), mp.nstr(exp_contour.acc.mesh.x_hi, 6))
    for a, sigma, mu in config.GEV_TEST_PARAMS:
        for p in config.GG_GEV_PROB_LEVELS:
            q_contour = gg_gev.gev_quantile(exp_contour, p, a=a, sigma=sigma, mu=mu,
                                             dps=config.CONTOUR_DPS)
            ref = gg_gev.reference_quantile_gev(p, a=a, sigma=sigma, mu=mu,
                                                 dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q_contour, ref))
            q_scipy = gg_gev.genextreme_sanity_check(p, a=a, sigma=sigma, mu=mu)
            gev_rows.append(dict(
                a=a, sigma=sigma, mu=mu, p=p,
                Q_GEV=float(q_contour),
                quantile_closed_form=float(ref),
                quantile_scipy_genextreme=q_scipy,
                rel_error=err,
            ))
    gev_df = pd.DataFrame(gev_rows)
    gev_df.to_csv("results/exp3_gev_quantiles.csv", index=False)
    log.info("Wrote results/exp3_gev_quantiles.csv (%d rows) in %.1fs",
              len(gev_df), time.time() - t0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for (a, c), grp in gg_df.groupby(["a", "c"]):
        axes[0].semilogy(grp["p"], grp["rel_error"], marker="o", label=f"a={a}, c={c}")
    axes[0].set_xlabel("probability level p")
    axes[0].set_ylabel("relative error")
    axes[0].set_title("GG quantile vs. gengamma-seeded Newton reference")
    axes[0].legend(fontsize=7)

    for (a, sigma, mu), grp in gev_df.groupby(["a", "sigma", "mu"]):
        axes[1].semilogy(grp["p"], grp["rel_error"], marker="o", label=f"a={a}, sigma={sigma}, mu={mu}")
    axes[1].set_xlabel("probability level p")
    axes[1].set_ylabel("relative error")
    axes[1].set_title("GEV quantile vs. elementary closed form")
    axes[1].legend(fontsize=7)

    fig.suptitle("exp_3: GG and GEV contour-integral quantile errors")
    fig.tight_layout()
    fig.savefig("results/exp3_gg_gev_errors.png", dpi=150)
    plt.close(fig)

    max_err_gg = gg_df["rel_error"].max()
    max_err_gev = gev_df["rel_error"].max()
    log.info("Max relative error: GG %.3e, GEV %.3e", max_err_gg, max_err_gev)
    return gg_df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()
