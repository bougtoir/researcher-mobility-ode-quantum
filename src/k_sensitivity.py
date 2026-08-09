#!/usr/bin/env python3
"""
k-sensitivity analysis for the endogenous ODE model.

For each civilisation we scale the observed k (number of distinct domestic
last-author groups needed for a viable community) by a set of multipliers and
recompute T = D + H_D + P_D and the margin T - M.  This shows how sensitive
the point-of-no-return story is to the threshold definition.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import ode_model_endogenous as odm


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "k_sensitivity"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--multipliers",
        type=float,
        nargs="+",
        default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        help="Factors applied to observed k per civilisation.",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.5,
        help="Fraction of r_critical used for the PI reproduction rate.",
    )
    args = parser.parse_args()

    a2g = odm.load_group_mapping()
    stats = odm.compute_coauthor_stats(a2g)

    cohort = pd.read_csv(odm.COHORT_DIR / "cohort.csv")
    rates = pd.read_csv(odm.COHORT_DIR / "transition_rates.csv").set_index("group")

    rows = []
    for mult in args.multipliers:
        k_override = {
            g: max(1, int(round(st.get("k_groups", 1) * mult)))
            for g, st in stats.items()
        }
        summary, _, _ = odm.run_endogenous_model(
            k_override=k_override,
            save=False,
            safety_factor=args.safety_factor,
            cohort_df=cohort,
            rates_df=rates,
        )
        for _, r in summary.iterrows():
            rows.append({
                "group": r["group"],
                "k_multiplier": mult,
                "k_used": r["k_used"],
                "k_observed": r["k_groups_observed"],
                "c_bar": r["c_bar"],
                "M_threshold": r["M_threshold"],
                "T_equilibrium": r["T_equilibrium"],
                "P_D_equilibrium": r["P_D_equilibrium"],
                "margin_to_threshold_T": r["margin_to_threshold_T"],
            })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "k_sensitivity.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR / 'k_sensitivity.csv'}")


if __name__ == "__main__":
    main()
