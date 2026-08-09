#!/usr/bin/env python3
"""
Bootstrap confidence intervals for the endogenous ODE equilibrium.

Resamples authors within each civilisation with replacement, recomputes
transition rates and annual inflows, and reruns the endogenous ODE.  The
resulting distribution of T_equilibrium, P_D_equilibrium and the threshold
margin is summarised with percentile intervals.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ode_model_endogenous as odm
from cohort_extraction import estimate_rates


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "bootstrap_ci"


def bootstrap_cohort(cohort, rng):
    """Stratified resample with replacement within origin_group."""
    pieces = []
    for _, g in cohort.groupby("origin_group"):
        n = len(g)
        pieces.append(g.sample(n=n, replace=True, random_state=rng.integers(2**32)))
    return pd.concat(pieces, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Number of bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--safety-factor", type=float, default=0.5)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cohort = pd.read_csv(odm.COHORT_DIR / "cohort.csv")

    # Coauthor statistics depend on the original work sample, not the bootstrap
    # resample, so compute them once outside the loop.
    a2g = odm.load_group_mapping()
    stats = odm.compute_coauthor_stats(a2g)

    rows = []
    for b in range(args.n_boot):
        boot = bootstrap_cohort(cohort, rng)
        rates = estimate_rates(boot).set_index("group")
        summary, _, _ = odm.run_endogenous_model(
            save=False,
            safety_factor=args.safety_factor,
            cohort_df=boot,
            rates_df=rates,
            a2g=a2g,
            stats=stats,
            compute_sensitivity=False,
            compute_pnr=False,
        )
        for _, r in summary.iterrows():
            rows.append({
                "bootstrap": b,
                "group": r["group"],
                "T_equilibrium": r["T_equilibrium"],
                "P_D_equilibrium": r["P_D_equilibrium"],
                "M_threshold": r["M_threshold"],
                "margin_to_threshold_T": r["margin_to_threshold_T"],
                "k_used": r["k_used"],
            })
        if (b + 1) % 20 == 0:
            print(f"Completed {b + 1}/{args.n_boot} bootstrap replicates")

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "bootstrap_draws.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    for g, gdf in df.groupby("group"):
        summary_rows.append({
            "group": g,
            "T_equilibrium_mean": gdf["T_equilibrium"].mean(),
            "T_equilibrium_median": gdf["T_equilibrium"].median(),
            "T_equilibrium_q025": gdf["T_equilibrium"].quantile(0.025),
            "T_equilibrium_q975": gdf["T_equilibrium"].quantile(0.975),
            "P_D_equilibrium_mean": gdf["P_D_equilibrium"].mean(),
            "P_D_equilibrium_q025": gdf["P_D_equilibrium"].quantile(0.025),
            "P_D_equilibrium_q975": gdf["P_D_equilibrium"].quantile(0.975),
            "margin_mean": gdf["margin_to_threshold_T"].mean(),
            "margin_q025": gdf["margin_to_threshold_T"].quantile(0.025),
            "margin_q975": gdf["margin_to_threshold_T"].quantile(0.975),
            "n": len(gdf),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS_DIR / "bootstrap_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
