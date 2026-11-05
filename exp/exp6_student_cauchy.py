"""exp_6: Student's t and Cauchy quantile construction with exact sanity checks.

For every degree of freedom `a` in `src.config.STUDENT_T_TEST_DOF` and every
probability level in `STUDENT_T_PROB_LEVELS`, builds the p-independent
Student's t(a) contour once (see `src/student_cauchy.py`) and evaluates the
quantile via the residue form (Eq. 4) with the bounded arctan observable,
comparing against a SciPy-seeded Newton-method reference (Appendix D.1).
Writes `results/exp6_student_t_quantiles.csv` and `results/exp6_student_t_errors.png`.

Separately, cross-validates the general contour method against the paper's
two EXACT closed forms (a=1 Cauchy, a=2) and SciPy, writing
`results/exp6_sanity_checks.csv`.
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
from src import student_cauchy as sc

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("exp6_student_cauchy")


def run() -> pd.DataFrame:
    rows = []
    t0 = time.time()
    for a in config.STUDENT_T_TEST_DOF:
        t_a0 = time.time()
        contour = sc.build_student_t_contour(a, dps=config.CONTOUR_DPS, max_cells_per_side=200, safety=1.25)
        log.info("Student-t(a=%s) contour built: mesh nodes=%d, x_hi=%s (%.1fs)",
                  a, len(contour.acc.mesh.xs), mp.nstr(contour.acc.mesh.x_hi, 6), time.time() - t_a0)
        for p in config.STUDENT_T_PROB_LEVELS:
            q = sc.student_t_quantile(contour, p, dps=config.CONTOUR_DPS)
            ref = sc.reference_quantile_student_t(a, p, dps=config.NEWTON_REFERENCE_DPS)
            err = float(cc.relative_error(q, ref)) if ref != 0 else float(abs(q))
            rows.append(dict(a=a, p=p, quantile=float(q), quantile_reference=float(ref), rel_error=err))
        log.info("Student-t(a=%s) done (%.1fs total)", a, time.time() - t_a0)
    df = pd.DataFrame(rows)
    df.to_csv("results/exp6_student_t_quantiles.csv", index=False)
    log.info("Wrote results/exp6_student_t_quantiles.csv (%d rows) in %.1fs", len(df), time.time() - t0)

    # --- Exact sanity checks: Cauchy (a=1) and Student a=2 ---
    from scipy.stats import cauchy as scipy_cauchy, t as scipy_t
    sanity_rows = []
    contour_1 = sc.build_student_t_contour(1.0, dps=config.CONTOUR_DPS, max_cells_per_side=200, safety=1.25)
    contour_2 = sc.build_student_t_contour(2.0, dps=config.CONTOUR_DPS, max_cells_per_side=200, safety=1.25)
    for p in [0.01, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 0.99]:
        q_c1 = sc.student_t_quantile(contour_1, p, dps=config.CONTOUR_DPS)
        q_c1_exact = sc.cauchy_exact_quantile(p)
        q_c1_scipy = scipy_cauchy.ppf(p)
        q_c2 = sc.student_t_quantile(contour_2, p, dps=config.CONTOUR_DPS)
        q_c2_exact = sc.student_t2_exact_quantile(p)
        q_c2_scipy = scipy_t.ppf(p, 2)
        sanity_rows.append(dict(
            case="cauchy(a=1)", p=p,
            quantile_contour=float(q_c1), quantile_exact=float(q_c1_exact), quantile_scipy=float(q_c1_scipy),
            rel_error_contour_vs_exact=float(cc.relative_error(q_c1, q_c1_exact)) if p != 0.5 else float(abs(q_c1)),
            rel_error_exact_vs_scipy=abs((float(q_c1_exact) - q_c1_scipy) / q_c1_scipy) if q_c1_scipy != 0 else abs(float(q_c1_exact)),
        ))
        sanity_rows.append(dict(
            case="student_t(a=2)", p=p,
            quantile_contour=float(q_c2), quantile_exact=float(q_c2_exact), quantile_scipy=float(q_c2_scipy),
            rel_error_contour_vs_exact=float(cc.relative_error(q_c2, q_c2_exact)) if p != 0.5 else float(abs(q_c2)),
            rel_error_exact_vs_scipy=abs((float(q_c2_exact) - q_c2_scipy) / q_c2_scipy) if q_c2_scipy != 0 else abs(float(q_c2_exact)),
        ))
    sanity_df = pd.DataFrame(sanity_rows)
    sanity_df.to_csv("results/exp6_sanity_checks.csv", index=False)
    log.info("Wrote results/exp6_sanity_checks.csv (%d rows)", len(sanity_df))
    log.info("Sanity checks: max rel_error contour-vs-exact=%.3e, exact-vs-scipy=%.3e",
              sanity_df["rel_error_contour_vs_exact"].max(), sanity_df["rel_error_exact_vs_scipy"].max())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a, grp in df.groupby("a"):
        ax.semilogy(grp["p"], grp["rel_error"], marker="o", label=f"a={a}")
    ax.set_xlabel("probability level p")
    ax.set_ylabel("relative error (residue form, Eq. 4, arctan observable)")
    ax.set_title("exp_6: Student's t / Cauchy contour-integral quantile vs. Newton reference")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("results/exp6_student_t_errors.png", dpi=150)
    plt.close(fig)

    max_err = df["rel_error"].max()
    log.info("Max relative error across all (a, p): %.3e", max_err)
    return df


if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    run()
