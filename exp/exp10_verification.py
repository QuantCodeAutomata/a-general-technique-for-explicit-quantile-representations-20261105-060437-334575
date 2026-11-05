"""exp_10: Numerical verification of distribution quantiles via contour integrals.

Evaluates the residue-form contour-integral quantile (Eq. 4) for the twelve
working laws of ``config.VERIFICATION_LAWS`` at the eight probability levels
of ``config.VERIFICATION_PROB_LEVELS``, compares against independently
computed Newton-method reference quantiles, and records the control checks
(accumulated CDF mass, real-density normalization) described in the
methodology of exp_10.
"""
from __future__ import annotations

import logging
import os

import mpmath as mp
import pandas as pd

from src import config
from src.verification import build_law
from src.contour_core import relative_error

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run() -> pd.DataFrame:
    rows = []
    control_rows = []

    for law in config.VERIFICATION_LAWS:
        logger.info("Building contour for working law %s", law["name"])
        handle = build_law(law)

        for mass, expected in zip(handle.accumulated_masses, handle.expected_masses):
            mass_real = mp.re(mass)
            mass_err = abs(mass - expected)
            control_rows.append({
                "law": handle.name,
                "check": "accumulated_cdf_mass",
                "value": float(mass_real),
                "expected": float(expected),
                "abs_error": float(mass_err),
            })
            logger.info("  accumulated mass %.6f vs expected %.6f (abs err %.3e)",
                        float(mass_real), float(expected), float(mass_err))

        density_err = abs(handle.density_integral - handle.expected_density_integral)
        control_rows.append({
            "law": handle.name,
            "check": "density_normalization",
            "value": float(handle.density_integral),
            "expected": float(handle.expected_density_integral),
            "abs_error": float(density_err),
        })
        logger.info("  density integral %.10f (abs err %.3e)", handle.density_integral, density_err)

        for p in config.VERIFICATION_PROB_LEVELS:
            q_contour = handle.quantile_fn(p)
            q_reference = handle.reference_fn(p)
            rel_err = relative_error(q_contour, q_reference)
            rows.append({
                "law": handle.name,
                "p": p,
                "q_contour": float(q_contour),
                "q_reference": float(q_reference),
                "relative_error": float(rel_err),
            })
            logger.info("  p=%.2f  Q_contour=%.10g  Q_reference=%.10g  rel_err=%.3e",
                        p, q_contour, q_reference, rel_err)

    df = pd.DataFrame(rows)
    control_df = pd.DataFrame(control_rows)
    return df, control_df


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    quantile_df, control_df = run()
    quantile_df.to_csv("results/exp10_verification_quantiles.csv", index=False)
    control_df.to_csv("results/exp10_verification_controls.csv", index=False)

    max_err = quantile_df["relative_error"].max()
    n_below_target = (quantile_df["relative_error"] < config.EXPECTED_MAX_RELATIVE_ERROR).sum()
    logger.info("Max relative error across all %d (law, p) combinations: %.3e",
                len(quantile_df), max_err)
    logger.info("%d / %d combinations below the paper's 1e-17 target",
                n_below_target, len(quantile_df))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 7))
    for law_name, group in quantile_df.groupby("law"):
        ax.semilogy(group["p"], group["relative_error"], marker="o", label=law_name)
    ax.axhline(config.EXPECTED_MAX_RELATIVE_ERROR, color="red", linestyle="--",
               label="paper target (1e-17)")
    ax.set_xlabel("probability level p")
    ax.set_ylabel("relative error |Q_contour - Q_reference| / |Q_reference|")
    ax.set_title("exp_10: contour-integral quantile relative errors across twelve working laws")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig("results/exp10_verification_errors.png", dpi=150)
    logger.info("Wrote results/exp10_verification_quantiles.csv, "
                "results/exp10_verification_controls.csv, "
                "results/exp10_verification_errors.png")
