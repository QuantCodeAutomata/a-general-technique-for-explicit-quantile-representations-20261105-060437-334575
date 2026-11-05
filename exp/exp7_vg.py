"""exp_7: Variance Gamma (VG) quantile construction with half-axis splitting.

For every (a, d) in `src.config.VG_TEST_PARAMS`, builds the two p-independent
half-axis contours once (`acc_plus`, `acc_minus`, Theorem A.2) and the
closed-form boundary probability `p_0` (Beta representation), then evaluates
the quantile at every level in `VG_PROB_LEVELS` plus `p_0` itself (to verify
Q(p_0) = 0 exactly), comparing each against a Newton-method reference
quantile (Appendix D.1). Writes `results/exp7_vg_quantiles.csv` and
`results/exp7_vg_errors.png`.

Known limitation (documented in `src/vg.py`'s module docstring and in
`tests/test_vg.py`): for a < 0 the fixed left cutoff x_lo=1e-14 used for both
half-axis contours (as the paper specifies explicitly for VG) leaves a
truncation-bias quantile-error floor of order 1e-6 (verified directly; two
alternative fixes were tried and both empirically regressed accuracy, see
src/vg.py), so the a=-0.3 case in this experiment is expected to show
materially larger relative errors than the other two (well-behaved, a>=0)
cases.
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
from src import vg

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp7_vg")


def run() -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for a, d in config.VG_TEST_PARAMS:
        t_build = time.time()
        contour = vg.build_vg_contour(a, d, dps=config.CONTOUR_DPS)
        log.info(
            "VG contour built: a=%s d=%s, p0=%s, x_hi_plus=%s, x_hi_minus=%s, "
            "width_factor=%s, mesh nodes=%d, build time=%.1fs",
            a, d, mp.nstr(contour.p0, 10),
            mp.nstr(contour.x_hi_plus, 6), mp.nstr(contour.x_hi_minus, 6),
            mp.nstr(contour.width_factor, 6), len(contour.acc_plus.mesh.xs),
            time.time() - t_build,
        )
        p0_float = float(contour.p0)
        p_levels = sorted(set(config.VG_PROB_LEVELS) | {p0_float})
        for p in p_levels:
            q = vg.vg_quantile(contour, p, dps=config.CONTOUR_DPS)
            is_p0 = abs(p - p0_float) < 1e-12
            if is_p0:
                ref = mp.mpf(0)
                err = cc.relative_error(q, ref) if ref != 0 else abs(q)
            else:
                ref = vg.reference_quantile_vg(a, d, p, dps=config.NEWTON_REFERENCE_DPS)
                err = cc.relative_error(q, ref)
            rows.append(dict(
                a=a, d=d, p=p, is_p0=is_p0,
                quantile=float(q),
                quantile_reference=float(ref),
                rel_error=float(err),
            ))
            log.info("  p=%.4f%s: Q=%s ref=%s relerr=%.3e",
                      p, " (=p0)" if is_p0 else "", mp.nstr(q, 10), mp.nstr(ref, 10), float(err))
    df = pd.DataFrame(rows)
    df.to_csv("results/exp7_vg_quantiles.csv", index=False)
    log.info("Wrote results/exp7_vg_quantiles.csv (%d rows) in %.1fs", len(df), time.time() - t0)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (a, d), grp in df.groupby(["a", "d"]):
        grp_nonzero = grp[grp["rel_error"] > 0]
        ax.semilogy(grp_nonzero["p"], grp_nonzero["rel_error"], marker="o", label=f"a={a}, d={d}")
    ax.set_xlabel("probability level p")
    ax.set_ylabel("relative error (residue form)")
    ax.set_title("exp_7: VG half-axis contour-integral quantile vs. Newton reference")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("results/exp7_vg_errors.png", dpi=150)
    plt.close(fig)

    for (a, d), grp in df.groupby(["a", "d"]):
        log.info("(a=%s, d=%s): max relative error = %.3e", a, d, grp["rel_error"].max())
    return df


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    run()
