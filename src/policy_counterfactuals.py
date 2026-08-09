#!/usr/bin/env python3
"""
Policy counterfactual simulation suite for the endogenous ODE model.

For each civilisation we apply proportional shocks to individual transition
rates (alpha, beta, h_D, h_A, p_D, p_A, d) and, for selected groups, small
multi-lever policy packages.  The output ranks interventions by their effect
on the domestic active pool T = D + H_D + P_D and on the PI pool P_D, and by
how much they move the community away from its point-of-no-return threshold
M = k * c_bar.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ode_model_endogenous as odm


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "policy_counterfactuals"

LEVER_RATES = ["alpha", "beta", "h_D", "h_A", "p_D", "p_A", "d"]
DEFAULT_FACTORS = [0.90, 0.95, 1.05, 1.10, 1.20]


def run_counterfactual(cohort, rates, safety_factor, group, lever, factor, a2g=None, stats=None):
    rates_cf = rates.copy()
    if lever in rates_cf.columns:
        rates_cf.loc[group, lever] = rates_cf.loc[group, lever] * factor
    summary, _, _ = odm.run_endogenous_model(
        save=False, safety_factor=safety_factor, cohort_df=cohort, rates_df=rates_cf,
        a2g=a2g, stats=stats,
        compute_sensitivity=False, compute_pnr=False,
    )
    row = summary[summary["group"] == group].iloc[0]
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--safety-factor", type=float, default=0.5)
    parser.add_argument("--factors", type=float, nargs="+", default=DEFAULT_FACTORS)
    parser.add_argument("--packages", action="store_true",
                        help="Also run bundled multi-lever packages for the smallest-margin groups.")
    args = parser.parse_args()

    cohort = pd.read_csv(odm.COHORT_DIR / "cohort.csv")
    rates = pd.read_csv(odm.COHORT_DIR / "transition_rates.csv").set_index("group")

    a2g = odm.load_group_mapping()
    stats = odm.compute_coauthor_stats(a2g)
    base_summary, _, _ = odm.run_endogenous_model(
        save=False, safety_factor=args.safety_factor,
        cohort_df=cohort, rates_df=rates, a2g=a2g, stats=stats,
        compute_sensitivity=False, compute_pnr=False,
    )
    base_summary = base_summary.set_index("group")

    rows = []
    for group in rates.index:
        base = base_summary.loc[group]
        for lever in LEVER_RATES:
            for f in args.factors:
                cf = run_counterfactual(cohort, rates, args.safety_factor, group, lever, f, a2g=a2g, stats=stats)
                rows.append({
                    "group": group,
                    "lever": lever,
                    "factor": f,
                    "baseline_T": base["T_equilibrium"],
                    "counterfactual_T": cf["T_equilibrium"],
                    "delta_T": cf["T_equilibrium"] - base["T_equilibrium"],
                    "delta_T_pct": 100.0 * (cf["T_equilibrium"] - base["T_equilibrium"]) / max(base["T_equilibrium"], 1e-12),
                    "baseline_margin": base["margin_to_threshold_T"],
                    "counterfactual_margin": cf["margin_to_threshold_T"],
                    "delta_margin": cf["margin_to_threshold_T"] - base["margin_to_threshold_T"],
                    "baseline_P_D": base["P_D_equilibrium"],
                    "counterfactual_P_D": cf["P_D_equilibrium"],
                    "delta_P_D": cf["P_D_equilibrium"] - base["P_D_equilibrium"],
                })

    # Optional multi-lever packages for the three smallest-margin groups
    if args.packages:
        sorted_groups = base_summary.sort_values("margin_to_threshold_T").index[:3]
        packages = [
            ("retention", {"d": 0.90, "beta": 1.10}),
            ("pi_pipeline", {"h_D": 1.10, "p_D": 1.10}),
            ("return_plus_retention", {"d": 0.90, "beta": 1.10, "p_D": 1.10}),
        ]
        for group in sorted_groups:
            base = base_summary.loc[group]
            for name, changes in packages:
                rates_cf = rates.copy()
                for lever, f in changes.items():
                    rates_cf.loc[group, lever] = rates_cf.loc[group, lever] * f
                summary, _, _ = odm.run_endogenous_model(
                    save=False, safety_factor=args.safety_factor,
                    cohort_df=cohort, rates_df=rates_cf, a2g=a2g, stats=stats,
                    compute_sensitivity=False, compute_pnr=False,
                )
                cf = summary[summary["group"] == group].iloc[0]
                rows.append({
                    "group": group,
                    "lever": f"package:{name}",
                    "factor": np.nan,
                    "baseline_T": base["T_equilibrium"],
                    "counterfactual_T": cf["T_equilibrium"],
                    "delta_T": cf["T_equilibrium"] - base["T_equilibrium"],
                    "delta_T_pct": 100.0 * (cf["T_equilibrium"] - base["T_equilibrium"]) / max(base["T_equilibrium"], 1e-12),
                    "baseline_margin": base["margin_to_threshold_T"],
                    "counterfactual_margin": cf["margin_to_threshold_T"],
                    "delta_margin": cf["margin_to_threshold_T"] - base["margin_to_threshold_T"],
                    "baseline_P_D": base["P_D_equilibrium"],
                    "counterfactual_P_D": cf["P_D_equilibrium"],
                    "delta_P_D": cf["P_D_equilibrium"] - base["P_D_equilibrium"],
                })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "counterfactuals.csv", index=False, encoding="utf-8-sig")

    # Rank interventions by normalised margin gain per 10% lever change
    ranked = []
    for g, gdf in df.groupby("group"):
        base_margin = base_summary.loc[g, "margin_to_threshold_T"]
        for _, r in gdf.iterrows():
            pct_lever = (r["factor"] - 1.0) * 100.0
            if pd.isna(pct_lever) or pct_lever == 0:
                continue
            abs_lever_change_10pct = abs(pct_lever) / 10.0
            ranked.append({
                "group": g,
                "lever": r["lever"],
                "direction": "increase" if pct_lever > 0 else "decrease",
                "lever_change_pct": pct_lever,
                "margin_gain": r["delta_margin"],
                "normalised_margin_gain_per_10pct": r["delta_margin"] / abs_lever_change_10pct,
            })
    ranked_df = pd.DataFrame(ranked).sort_values(["group", "normalised_margin_gain_per_10pct"], ascending=[True, False])
    ranked_df.to_csv(RESULTS_DIR / "ranked_interventions.csv", index=False, encoding="utf-8-sig")

    print("Top 3 interventions per group (margin gain per 10% lever change):")
    print(ranked_df.groupby("group").head(3).to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
