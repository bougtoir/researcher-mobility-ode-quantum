#!/usr/bin/env python3
"""
Network externalities across civilisations in the endogenous ODE.

The single-group model treats each civilisation as closed: domestic PI pool P_D
feeds back into domestic early-career entrants.  This script adds cross-group
spillovers: early-career entrants in group i are also attracted by the size of
the PI diaspora of group j currently abroad (P_A_j), weighted by co-authorship
and collaboration exposure inferred from the stratified OpenAlex sample.

A 6N x 6N coupled linear system is assembled and its equilibrium solved.  The
cross-group inflow term is:

    inflow_D_i = I0_i + r_i P_D_i + sum_{j != i} gamma_{ij} P_A_j

where gamma_{ij} = r_i * spillover_factor * W_{ij} / W_{ii} and W_{ij} is the
count of sampled AI/ML works on which institutions from groups i and j
co-occur (W_{ii} is the count of works with at least one i-affiliated author).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import ode_model_endogenous as odm


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "network_externalities"
SAMPLE_FILE = BASE_DIR / "data" / "cohort" / "raw_sampled_works.json"


def collaboration_weights(a2g):
    """Return normalised group x group collaboration matrix W / diag(W)."""
    with open(SAMPLE_FILE, "r", encoding="utf-8") as f:
        works = json.load(f)

    raw_counts = defaultdict(lambda: defaultdict(int))
    for w in works:
        groups = sorted(set(odm.work_groups(w, a2g)))
        for g in groups:
            raw_counts[g][g] += 1
        for i, g1 in enumerate(groups):
            for g2 in groups[i + 1:]:
                raw_counts[g1][g2] += 1
                raw_counts[g2][g1] += 1

    groups = sorted(raw_counts.keys())
    n = len(groups)
    W = np.zeros((n, n))
    for i, g in enumerate(groups):
        for j, h in enumerate(groups):
            W[i, j] = raw_counts[g][h]

    # Normalise cross terms by within-group count
    norm = np.zeros((n, n))
    for i in range(n):
        diag = W[i, i]
        if diag > 0:
            for j in range(n):
                norm[i, j] = W[i, j] / diag
    return groups, norm, W


def build_multi_group_matrix(groups, rates, inflows, r_dict, gamma_weight, spillover_factor):
    """Build and return the 6N x 6N coupled transition matrix."""
    n = len(groups)
    N = 6 * n
    M = np.zeros((N, N))
    for i, g in enumerate(groups):
        params = {c: float(rates.loc[g, c]) for c in odm.RATE_NAMES}
        params["r"] = r_dict[g]
        block = odm.build_matrix(params)
        M[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = block
        # Cross-group spillover into D_i from P_A_j (compartment 5 of group j)
        for j in range(n):
            if j == i:
                continue
            gamma = r_dict[g] * spillover_factor * gamma_weight[i, j]
            if gamma != 0.0:
                M[i * 6, j * 6 + 5] = gamma
    return M


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spillover-factor", type=float, default=0.1,
                        help="Scale factor for cross-group PI spillovers.")
    parser.add_argument("--safety-factor", type=float, default=0.5)
    args = parser.parse_args()

    a2g = odm.load_group_mapping()
    stats = odm.compute_coauthor_stats(a2g)

    cohort = pd.read_csv(odm.COHORT_DIR / "cohort.csv")
    rates = pd.read_csv(odm.COHORT_DIR / "transition_rates.csv").set_index("group")

    I0_dict, r_dict, rcrit_dict, _ = odm.estimate_endogenous_inflow(
        cohort, rates, safety_factor=args.safety_factor
    )

    all_groups, gamma_weight, W = collaboration_weights(a2g)
    # Keep only groups that have rates and inflow
    groups = [g for g in all_groups if g in rates.index and I0_dict.get(g, 0) > 0]

    rates = rates.loc[groups]
    idx_map = {g: i for i, g in enumerate(all_groups)}
    keep_idx = [idx_map[g] for g in groups]
    gamma_weight = gamma_weight[np.ix_(keep_idx, keep_idx)]

    M = build_multi_group_matrix(groups, rates, I0_dict, r_dict, gamma_weight, args.spillover_factor)
    max_eig = max(np.linalg.eigvals(M).real)

    b = np.zeros(len(groups) * 6)
    for i, g in enumerate(groups):
        b[i * 6] = -I0_dict[g]

    y_eq = np.linalg.solve(M, b)

    rows = []
    for i, g in enumerate(groups):
        d = y_eq[i * 6:(i + 1) * 6]
        T = d[0] + d[2] + d[4]
        st = stats.get(g, {})
        c_bar = st.get("mean_authors", np.nan)
        k = st.get("k_groups", 0)
        M_th = k * c_bar if not (np.isnan(c_bar) or c_bar == 0) else np.nan
        rows.append({
            "group": g,
            "spillover_factor": args.spillover_factor,
            "I0": I0_dict[g],
            "r": r_dict[g],
            "r_critical": rcrit_dict.get(g, np.nan),
            "c_bar": c_bar,
            "k_groups": k,
            "M_threshold": M_th,
            "T_equilibrium": T,
            "D_eq": d[0],
            "A_eq": d[1],
            "H_D_eq": d[2],
            "H_A_eq": d[3],
            "P_D_eq": d[4],
            "P_A_eq": d[5],
            "margin_to_threshold_T": T - M_th,
            "max_eigenvalue_real": max_eig,
        })

    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"network_eq_spillover_{args.spillover_factor}.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")

    # Save collaboration matrix for reference
    W_df = pd.DataFrame(W, index=all_groups, columns=all_groups)
    W_df.to_csv(RESULTS_DIR / "collaboration_matrix.csv", encoding="utf-8-sig")

    print(f"Spillover factor = {args.spillover_factor}, max real eigenvalue = {max_eig:.6f}")
    print(df[["group", "T_equilibrium", "M_threshold", "margin_to_threshold_T"]].to_string(index=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
