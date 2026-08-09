#!/usr/bin/env python3
"""
Behavioral layers: postdoc-stage vs mid-career abroad episodes.

The core cohort flags an author as "abroad" if any AI/ML work within the first
ABROAD_WINDOW years has an institution in a different civilisation.  This
script splits those abroad episodes by timing:

  - postdoc_abroad : first abroad year <= career_start + 2
  - mid_abroad     : first abroad year > career_start + 2 (but still <= window)

For each civilisation it estimates separate return rates (recent_group ==
origin_group) and PI attainment rates (pi == True) for the two layers.  It then
runs a counterfactual in which the postdoc return rate is improved (by a
configurable factor or by aligning it with the mid-career return rate) and
re-computes the endogenous ODE equilibrium, treating the single "abroad"
compartment as a weighted mixture of the two layers.

This is a step toward explicitly disaggregating the A compartment into
A_postdoc and A_senior in future model versions.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import ode_model_endogenous as odm


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "behavioral_layers"


def smooth_prop(successes, n, prior=1.0):
    """Laplace-smoothed proportion matching cohort_extraction.estimate_rates."""
    if n == 0:
        return 0.0
    return (successes + prior) / (n + 2 * prior)


def p_to_beta(p, horizon=10.0, rate_cap=2.0):
    """Convert a return probability to a constant annual hazard rate."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return rate_cap
    return min(-math.log(1.0 - p) / horizon, rate_cap)


def classify_abroad_layer(row):
    if not row["abroad"] or pd.isna(row["abroad_year"]):
        return None
    return "postdoc" if (row["abroad_year"] - row["career_start"]) <= 2 else "mid"


def layer_rates(cohort):
    cohort = cohort.copy()
    cohort["abroad_layer"] = cohort.apply(classify_abroad_layer, axis=1)
    rows = []
    for g, df in cohort.groupby("origin_group"):
        abroad = df[df["abroad"] == True]
        n_post = ((abroad["abroad_layer"] == "postdoc")).sum()
        n_mid = ((abroad["abroad_layer"] == "mid")).sum()

        def return_rate(sub):
            if len(sub) == 0:
                return 0.0
            return (sub["recent_group"] == sub["origin_group"]).mean()

        def pi_rate(sub):
            if len(sub) == 0:
                return 0.0
            return sub["pi"].mean()

        postdoc = abroad[abroad["abroad_layer"] == "postdoc"]
        mid = abroad[abroad["abroad_layer"] == "mid"]
        r_post = return_rate(postdoc)
        r_mid = return_rate(mid)
        pi_post = pi_rate(postdoc)
        pi_mid = pi_rate(mid)

        if len(abroad) > 0:
            beta_baseline = (n_post * r_post + n_mid * r_mid) / len(abroad)
        else:
            beta_baseline = 0.0

        rows.append({
            "group": g,
            "n_abroad": len(abroad),
            "n_postdoc_abroad": n_post,
            "n_mid_abroad": n_mid,
            "postdoc_return_rate": r_post,
            "mid_return_rate": r_mid,
            "postdoc_pi_rate": pi_post,
            "mid_pi_rate": pi_mid,
            "beta_baseline": beta_baseline,
        })
    return pd.DataFrame(rows)


def counterfactual_T(cohort, rates, group, beta_cf, a2g, stats):
    rates_cf = rates.copy()
    rates_cf.loc[group, "beta"] = beta_cf
    summary, _, _ = odm.run_endogenous_model(
        save=False,
        safety_factor=0.5,
        cohort_df=cohort,
        rates_df=rates_cf,
        a2g=a2g,
        stats=stats,
        compute_sensitivity=False,
        compute_pnr=False,
    )
    return summary.set_index("group").loc[group]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--postdoc-return-factor", type=float, default=1.10,
                        help="Multiplier applied to the observed postdoc return rate.")
    parser.add_argument("--cap-to-mid", action="store_true",
                        help="Cap the counterfactual postdoc return rate at the mid-career return rate.")
    args = parser.parse_args()

    cohort = pd.read_csv(odm.COHORT_DIR / "cohort.csv")
    rates = pd.read_csv(odm.COHORT_DIR / "transition_rates.csv").set_index("group")

    a2g = odm.load_group_mapping()
    stats = odm.compute_coauthor_stats(a2g)

    base_summary, _, _ = odm.run_endogenous_model(
        save=False, safety_factor=0.5,
        cohort_df=cohort, rates_df=rates, a2g=a2g, stats=stats,
        compute_sensitivity=False, compute_pnr=False,
    )
    base = base_summary.set_index("group")

    layers = layer_rates(cohort).set_index("group")

    rows = []
    for g in rates.index:
        if g not in layers.index or layers.loc[g, "n_abroad"] == 0:
            continue
        lr = layers.loc[g]
        n_post = lr["n_postdoc_abroad"]
        n_mid = lr["n_mid_abroad"]
        n_total = lr["n_abroad"]
        r_post_cf = lr["postdoc_return_rate"] * args.postdoc_return_factor
        if args.cap_to_mid and lr["mid_return_rate"] > 0:
            r_post_cf = min(r_post_cf, lr["mid_return_rate"])
        r_post_cf = min(max(r_post_cf, 0.0), 1.0)

        # Combine postdoc and mid-career return probabilities, apply the same
        # Laplace smoothing used in cohort_extraction.estimate_rates, then convert
        # to the hazard scale used for beta in transition_rates.csv.
        p_return_cf = (n_post * r_post_cf + n_mid * lr["mid_return_rate"]) / n_total if n_total > 0 else lr["beta_baseline"]
        p_return_cf = min(max(p_return_cf, 0.0), 1.0)
        p_return_cf_smooth = smooth_prop(p_return_cf * n_total, n_total)
        beta_cf = p_to_beta(p_return_cf_smooth)
        beta_baseline = rates.loc[g, "beta"]
        cf = counterfactual_T(cohort, rates, g, beta_cf, a2g, stats)
        rows.append({
            "group": g,
            "n_postdoc_abroad": n_post,
            "n_mid_abroad": n_mid,
            "postdoc_return_rate": lr["postdoc_return_rate"],
            "mid_return_rate": lr["mid_return_rate"],
            "beta_baseline": beta_baseline,
            "beta_counterfactual": beta_cf,
            "T_baseline": base.loc[g, "T_equilibrium"],
            "T_counterfactual": cf["T_equilibrium"],
            "delta_T": cf["T_equilibrium"] - base.loc[g, "T_equilibrium"],
            "delta_margin": cf["margin_to_threshold_T"] - base.loc[g, "margin_to_threshold_T"],
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "behavioral_layers_counterfactual.csv", index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
