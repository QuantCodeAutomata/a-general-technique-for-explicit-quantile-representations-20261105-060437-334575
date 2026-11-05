"""exp_4: Generalized hyperbolic (GH) quantile construction.

For every (a, b, d) in `src.config.GH_TEST_PARAMS` and every probability
level in `GH_PROB_LEVELS`, builds the p-independent GH(a, b, d) contour once
(constant contour height, Eq. A.5) and evaluates the working quantile with
the residue form (Eq. 4), comparing it against a SciPy/mpmath-quadrature
Newton-method reference (methodology, exp_4). Writes
`results/exp4_gh_quantiles.csv` and `results/exp4_gh_errors.png`.
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
from src import gh

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp4_gh")


def run() -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for a, b, d in config.GH_TEST_PARAMS:
        t_build = time.time()
        contour = gh.build_gh_contour(a, b, d, dps=config.CONTOUR_DPS)
        log.info("GH contour built: a=%s b=%s d=%s, h=%s, mesh nodes=%d, x_lo=%s, "
                  "x_hi=%s (%.1fs)",
                  a, b, d, mp.nstr(contour.h_const, 6), len(contour.acc.mesh.xs),
                  mp.nstr(contour.acc.mesh.x_lo, 6), mp.nstr(contour.acc.mesh.x_hi, 6),
                  time.time() - t_build)
        for p in config.GH_PROB_LEVELS:
            q = gh.gh_quantile(contour, p, dps=config.CONTOUR_DPS)
            ref = gh.reference_quantile_gh(a, b, d, p, dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q, ref))
            log.info("  p=%s quantile=%s reference=%s rel_error=%.3e",
                      p, mp.nstr(q, 10), mp.nstr(ref, 10), err)
            rows.append(dict(
                a=a, b=b, d=d, p=p,
                quantile=float(q),
                quantile_reference=float(ref),
                rel_error=err,
            ))
    df = pd.DataFrame(rows)
    df.to_csv("results/exp4_gh_quantiles.csv", index=False)
    log.info("Wrote results/exp4_gh_quantiles.csv (%d rows) in %.1fs", len(df), time.time() - t0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (a, b, d), grp in df.groupby(["a", "b", "d"]):
        ax.semilogy(grp["p"], grp["rel_error"], marker="o", label=f"a={a}, b={b}, d={d}")
    ax.set_xlabel("probability level p")
    ax.set_ylabel("relative error (residue form, Eq. 4)")
    ax.set_title("exp_4: GH contour-integral quantile vs. Newton reference")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/exp4_gh_errors.png", dpi=150)
    plt.close(fig)

    log.info("Max relative error: %.3e", df["rel_error"].max())
    return df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()
