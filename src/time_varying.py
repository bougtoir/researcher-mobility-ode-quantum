#!/usr/bin/env python3
"""
Time-varying ODE parameters: pre/post split of the AI/ML cohort.

Career start years are split at a configurable cutoff (default 2010).
Transition rates and inflow parameters are re-estimated for each period,
and the endogenous ODE equilibrium is recomputed.  This reveals whether a
civilisation's domestic research community has become more or less resilient
over time.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ode_model_endogenous as odm
from cohort_extraction import estimate_rates


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "time_varying"


def split_cohort(cohort, cutoff):
    """Return (early, late) cohorts by career_start."""
    early = cohort[cohort["career_start"] <= cutoff].copy()
    late = cohort[cohort["career_start"] > cutoff].copy()
    return early, late


def run_period(cohort, rates, period_label, safety_factor=0.5):
    """Run endogenous model for a given cohort/rates."""
    start_year = int(cohort["career_start"].min())
    end_year = int(cohort["career_start"].max())
    summary, sens, pnr = odm.run_endogenous_model(
        save=False,
        safety_factor=safety_factor,
        cohort_df=cohort,
        rates_df=rates,
        career_start_min=start_year,
        career_start_max=end_year,
    )
    summary["period"] = period_label
    sens["period"] = period_label
    pnr["period"] = period_label
    return summary, sens, pnr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=2010,
                        help="Career-start year that separates early/late periods.")
    parser.add_argument("--safety-factor", type=float, default=0.5)
    args = parser.parse_args()

    cohort = pd.read_csv(odm.COHORT_DIR / "cohort.csv")
    early, late = split_cohort(cohort, args.cutoff)

    if len(early) == 0 or len(late) == 0:
        raise ValueError("Empty early or late cohort after split")

    early_rates = estimate_rates(early).set_index("group")
    late_rates = estimate_rates(late).set_index("group")

    early_summary, early_sens, early_pnr = run_period(
        early, early_rates, "early", safety_factor=args.safety_factor
    )
    late_summary, late_sens, late_pnr = run_period(
        late, late_rates, "late", safety_factor=args.safety_factor
    )

    summary = pd.concat([early_summary, late_summary], ignore_index=True)
    sens = pd.concat([early_sens, late_sens], ignore_index=True)
    pnr = pd.concat([early_pnr, late_pnr], ignore_index=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "equilibrium_summary.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(RESULTS_DIR / "sensitivity.csv", index=False, encoding="utf-8-sig")
    pnr.to_csv(RESULTS_DIR / "point_of_no_return.csv", index=False, encoding="utf-8-sig")

    # Period comparison keyed by group
    compare = []
    for g in summary["group"].unique():
        e = early_summary[early_summary["group"] == g]
        l = late_summary[late_summary["group"] == g]
        if e.empty or l.empty:
            continue
        er = e.iloc[0]
        lr = l.iloc[0]
        compare.append({
            "group": g,
            "T_early": er["T_equilibrium"],
            "T_late": lr["T_equilibrium"],
            "delta_T": lr["T_equilibrium"] - er["T_equilibrium"],
            "pct_delta_T": 100.0 * (lr["T_equilibrium"] - er["T_equilibrium"]) / max(er["T_equilibrium"], 1e-12),
            "M_early": er["M_threshold"],
            "M_late": lr["M_threshold"],
            "margin_early": er["margin_to_threshold_T"],
            "margin_late": lr["margin_to_threshold_T"],
            "delta_margin": lr["margin_to_threshold_T"] - er["margin_to_threshold_T"],
        })
    compare_df = pd.DataFrame(compare)
    compare_df.to_csv(RESULTS_DIR / "period_comparison.csv", index=False, encoding="utf-8-sig")

    print("=== Period comparison ===")
    print(compare_df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
