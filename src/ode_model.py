#!/usr/bin/env python3
"""
Coupled per-civilisation compartmental ODE model for AI/ML researcher mobility.

For each civilisation we track:
  D   : domestic early-career researchers (not yet hit, not PI, not abroad)
  A   : abroad early-career researchers
  H_D : domestic "hit" researchers (top-10% cited, non-PI)
  H_A : abroad "hit" researchers
  P_D : domestic PIs
  P_A : abroad PIs
  L   : left academia (absorbing, not tracked explicitly)

The domestic active pool is T = D + H_D + P_D.

This first implementation keeps inflow exogenous; future extensions can make
inflow depend on P_D once PhD/mentorship linkage data are available.

Usage:
    python src/ode_model.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import solve


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
COHORT_DIR = DATA_DIR / "cohort"
RESULTS_DIR = BASE_DIR / "results"

COMPARTMENTS = ["D", "A", "H_D", "H_A", "P_D", "P_A"]
RATE_NAMES = ["alpha", "beta", "h_D", "h_A", "p_D", "p_A", "d"]


def load_group_mapping():
    with open(DATA_DIR / "country_civilization_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    a2g = {}
    for a3, info in mapping.items():
        a2 = info.get("alpha_2")
        g = info.get("group")
        if a2 and g:
            a2g[a2] = g
    return a2g


def work_groups(work, a2g):
    """Return the set of civilisation groups appearing on a work."""
    groups = set()
    for auth in work.get("authorships", []):
        for inst in auth.get("institutions", []):
            cc = inst.get("country_code")
            if cc in a2g:
                groups.add(a2g[cc])
    return groups


def last_author_ids(work):
    """Return the author id(s) whose position is 'last' on this work."""
    ids = []
    for auth in work.get("authorships", []):
        if auth.get("author_position") == "last":
            raw = (auth.get("author") or {}).get("id")
            if raw:
                ids.append(raw.split("/")[-1])
    return ids


def n_authors(work):
    return len(work.get("authorships", []))


def compute_coauthor_stats(a2g, recent_years=(2020, 2021, 2022, 2023)):
    """
    From the cached stratified work sample, compute for each civilisation:
      - mean_authors : mean number of authors per AI/ML work
      - k_groups     : median number of distinct last-author groups per recent year
    """
    sample_file = COHORT_DIR / "raw_sampled_works.json"
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample cache not found: {sample_file}. Run cohort_extraction.py first.")

    with open(sample_file, "r", encoding="utf-8") as f:
        works = json.load(f)

    group_works = defaultdict(list)
    for w in works:
        y = w.get("publication_year")
        for g in work_groups(w, a2g):
            group_works[g].append((w, y))

    stats = {}
    for g, items in group_works.items():
        n_auth_counts = [n_authors(w) for w, _ in items]
        c_bar = float(np.mean(n_auth_counts)) if n_auth_counts else np.nan

        yearly_last = defaultdict(set)
        for w, y in items:
            if y in recent_years:
                for lid in last_author_ids(w):
                    yearly_last[y].add(lid)
        yearly_counts = [len(yearly_last[y]) for y in recent_years if y in yearly_last]
        if not yearly_counts:
            yearly_last = defaultdict(set)
            for w, y in items:
                for lid in last_author_ids(w):
                    yearly_last[y].add(lid)
            yearly_counts = [len(s) for s in yearly_last.values()]
        k_groups = int(np.median(yearly_counts)) if yearly_counts else 10

        stats[g] = {
            "mean_authors": c_bar,
            "k_groups": k_groups,
            "n_sampled_works": len(items),
            "recent_last_author_counts": yearly_counts,
        }
    return stats


def estimate_inflows(cohort):
    """Annual new domestic entrants per group from observed career starts."""
    df = cohort[(cohort["career_start"] >= 2000) & (cohort["career_start"] <= 2016)]
    counts = (df.groupby(["origin_group", "career_start"])
                .size()
                .unstack(fill_value=0)
                .reindex(columns=range(2000, 2017), fill_value=0))
    annual_mean = counts.mean(axis=1)
    annual_mean = annual_mean.replace([np.inf, -np.inf], np.nan).fillna(0)
    return annual_mean.to_dict()


def build_matrix(p):
    """
    Build the 6x6 transition matrix M for a single civilisation such that
    dy/dt = M y + b,  b = [I, 0, 0, 0, 0, 0]^T.
    """
    alpha = p["alpha"]
    beta = p["beta"]
    h_D = p["h_D"]
    h_A = p["h_A"]
    p_D = p["p_D"]
    p_A = p["p_A"]
    d = p["d"]

    M = np.zeros((6, 6))
    # D row
    M[0, 0] = -(alpha + h_D + d)
    M[0, 1] = beta                       # A -> D (return)
    # A row
    M[1, 0] = alpha
    M[1, 1] = -(h_A + beta + d)
    # H_D row
    M[2, 0] = h_D
    M[2, 2] = -(p_D + d)
    M[2, 3] = beta                       # H_A -> H_D (return)
    # H_A row
    M[3, 1] = h_A
    M[3, 3] = -(p_A + beta + d)
    # P_D row
    M[4, 2] = p_D
    M[4, 4] = -d
    M[4, 5] = beta                       # P_A -> P_D (return)
    # P_A row
    M[5, 3] = p_A
    M[5, 5] = -(beta + d)
    return M


def equilibrium(params, inflow):
    """Solve M y_eq + b = 0; return vector and domestic active T = D + H_D + P_D."""
    M = build_matrix(params)
    b = np.zeros(6)
    b[0] = -inflow
    y_eq = solve(M, b, assume_a="gen")
    T = y_eq[0] + y_eq[2] + y_eq[4]
    return y_eq, T


def compartment_value(y_eq, name):
    """Return a compartment value or named aggregate."""
    if name == "T":
        return y_eq[0] + y_eq[2] + y_eq[4]
    if name == "P":
        return y_eq[4]
    return y_eq[COMPARTMENTS.index(name)]


def sensitivity_table(params, inflow, target="T", delta=0.01):
    """
    Compute the elasticity of the equilibrium target with respect to each
    transition rate and to the inflow I.
    """
    y0, _ = equilibrium(params, inflow)
    base = compartment_value(y0, target)
    rows = []
    rates_to_test = RATE_NAMES + ["I"]
    for rate_name in rates_to_test:
        if rate_name == "I":
            I2 = inflow * (1 + delta)
            y2, _ = equilibrium(params, I2)
        else:
            p2 = dict(params)
            p2[rate_name] = params[rate_name] * (1 + delta)
            y2, _ = equilibrium(p2, inflow)
        val2 = compartment_value(y2, target)
        elasticity = ((val2 - base) / base) / delta if base != 0 and delta != 0 else np.nan
        rows.append({
            "rate": rate_name,
            "value": inflow if rate_name == "I" else params[rate_name],
            "target_value_after_1pct_change": val2,
            "elasticity": elasticity,
        })
    return pd.DataFrame(rows).sort_values("elasticity", ascending=False)


def point_of_no_return(params, inflow, M_threshold, rate_name, target="T",
                       min_factor=0.001, max_factor=20.0, n_points=2000):
    """
    Find the multiplicative factor (relative to the current rate) at which the
    equilibrium target equals M_threshold.  Reports whether the crossing is
    within the scanned range and the direction of the effect.
    """
    if rate_name == "I":
        current = inflow
    else:
        current = params[rate_name]
    if current <= 0:
        # vary additively from 0
        test_values = np.linspace(0.0, max_factor * 0.05, n_points)
    else:
        test_values = current * np.linspace(min_factor, max_factor, n_points)
    test_values = np.unique(np.maximum(test_values, 1e-12))

    def target_at(v):
        if rate_name == "I":
            y, _ = equilibrium(params, v)
        else:
            p2 = dict(params)
            p2[rate_name] = v
            y, _ = equilibrium(p2, inflow)
        return compartment_value(y, target)

    Ts = np.array([target_at(v) for v in test_values])
    diff = Ts - M_threshold

    # Determine monotonic direction from the scanned extremes
    if Ts[-1] > Ts[0]:
        direction = +1
    elif Ts[-1] < Ts[0]:
        direction = -1
    else:
        direction = 0

    current_T = target_at(current)
    if current_T <= M_threshold:
        return {
            "rate_name": rate_name,
            "current_rate": current,
            "critical_factor": np.nan,
            "critical_rate": np.nan,
            "T_at_critical": current_T,
            "is_within_bounds": False,
            "direction": direction,
            "note": "already_below_threshold",
        }

    # Find first crossing of M_threshold as the rate is varied from current
    # toward the bound that decreases T.
    if direction == 0:
        return {
            "rate_name": rate_name,
            "current_rate": current,
            "critical_factor": np.nan,
            "critical_rate": np.nan,
            "T_at_critical": current_T,
            "is_within_bounds": False,
            "direction": 0,
            "note": "no_effect",
        }

    # Monotonic scan: as we move from min_factor to max_factor, T either always
    # increases or always decreases.  The first crossing we encounter when moving
    # *from the current factor toward the side that lowers T* is the relevant one.
    current_factor = 1.0
    factor_values = test_values / current if current > 0 else np.linspace(0, max_factor, n_points)
    if direction == -1:
        # increasing rate lowers T; scan from current_factor upward
        idx_mask = factor_values >= current_factor
        scan_factors = factor_values[idx_mask]
        scan_diff = diff[idx_mask]
    else:
        # increasing rate raises T; scan from current_factor downward
        idx_mask = factor_values <= current_factor
        scan_factors = factor_values[idx_mask][::-1]
        scan_diff = diff[idx_mask][::-1]

    crossings = np.where(np.diff(np.sign(scan_diff)))[0]
    if len(crossings) == 0:
        # No crossing in this direction within bounds
        last_factor = float(scan_factors[-1])
        last_rate = last_factor * current if current > 0 else last_factor
        return {
            "rate_name": rate_name,
            "current_rate": current,
            "critical_factor": last_factor,
            "critical_rate": last_rate,
            "T_at_critical": target_at(last_rate),
            "is_within_bounds": False,
            "direction": direction,
            "note": "no_crossing_in_scanned_range",
        }

    idx = crossings[0]
    f0, f1 = scan_factors[idx], scan_factors[idx + 1]
    t0, t1 = scan_diff[idx] + M_threshold, scan_diff[idx + 1] + M_threshold
    # linear interpolation in factor space, using target values
    frac = (M_threshold - t0) / (t1 - t0) if t1 != t0 else 0.0
    critical_factor = f0 + frac * (f1 - f0)
    critical_rate = critical_factor * current if current > 0 else critical_factor
    return {
        "rate_name": rate_name,
        "current_rate": current,
        "critical_factor": critical_factor,
        "critical_rate": critical_rate,
        "T_at_critical": M_threshold,
        "is_within_bounds": True,
        "direction": direction,
        "note": "crossing_found",
    }


def run_model(k_override=None, save=True, scan_targets=("T", "P")):
    """
    Load rates and cohort, compute equilibrium, sensitivities, and
    point-of-no-return thresholds for each civilisation.
    """
    a2g = load_group_mapping()
    stats = compute_coauthor_stats(a2g)

    cohort = pd.read_csv(COHORT_DIR / "cohort.csv")
    rates = pd.read_csv(COHORT_DIR / "transition_rates.csv").set_index("group")

    inflows = estimate_inflows(cohort)

    rate_cols = RATE_NAMES.copy()
    summary_rows = []
    sensitivity_frames = []
    pnr_frames = []

    for group, row in rates.iterrows():
        params = {c: float(row[c]) for c in rate_cols}
        I = inflows.get(group, 0.0)
        if I <= 0 or np.isnan(I):
            continue

        st = stats.get(group, {})
        c_bar = st.get("mean_authors", np.nan)
        k_obs = st.get("k_groups", 0)
        k = k_override if k_override is not None else k_obs
        M_threshold = k * c_bar if not (np.isnan(c_bar) or c_bar == 0) else np.nan

        y_eq, T_eq = equilibrium(params, I)
        P_eq = y_eq[4]

        summary_rows.append({
            "group": group,
            "inflow_I": I,
            "c_bar": c_bar,
            "k_groups_observed": k_obs,
            "k_used": k,
            "M_threshold": M_threshold,
            "T_equilibrium": T_eq,
            "D_eq": y_eq[0],
            "A_eq": y_eq[1],
            "H_D_eq": y_eq[2],
            "H_A_eq": y_eq[3],
            "P_D_eq": y_eq[4],
            "P_A_eq": y_eq[5],
            "margin_to_threshold_T": T_eq - M_threshold,
            "P_D_equilibrium": P_eq,
        })

        for target in scan_targets:
            target_label = {"T": "domestic_active", "P": "domestic_PIs"}.get(target, target)
            # Sensitivity
            sens = sensitivity_table(params, I, target=target)
            sens["group"] = group
            sens["target"] = target_label
            sens["M_threshold"] = M_threshold
            sensitivity_frames.append(sens)

            # Point of no return
            for rate_name in rate_cols + ["I"]:
                if rate_name != "I" and params[rate_name] <= 0:
                    continue
                pnr = point_of_no_return(params, I, M_threshold, rate_name, target=target)
                pnr["group"] = group
                pnr["target"] = target_label
                pnr["T_equilibrium"] = T_eq
                pnr["M_threshold"] = M_threshold
                pnr_frames.append(pnr)

    summary_df = pd.DataFrame(summary_rows)
    sens_df = pd.concat(sensitivity_frames, ignore_index=True)
    pnr_df = pd.DataFrame(pnr_frames)

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(RESULTS_DIR / "equilibrium_summary.csv", index=False, encoding="utf-8-sig")
        sens_df.to_csv(RESULTS_DIR / "sensitivity.csv", index=False, encoding="utf-8-sig")
        pnr_df.to_csv(RESULTS_DIR / "point_of_no_return.csv", index=False, encoding="utf-8-sig")

    return summary_df, sens_df, pnr_df


def summarize_results():
    """Generate human-readable intervention summary from baseline results."""
    summary = pd.read_csv(RESULTS_DIR / "equilibrium_summary.csv")
    sens = pd.read_csv(RESULTS_DIR / "sensitivity.csv")
    pnr = pd.read_csv(RESULTS_DIR / "point_of_no_return.csv")

    TRANSITIONS = ["alpha", "beta", "h_D", "h_A", "p_D", "p_A", "d", "I"]

    def topn(df, target, rates, n=3):
        d = df[(df["target"] == target) & (df["rate"].isin(rates))].copy()
        d["abs_elasticity"] = d["elasticity"].abs()
        d = d.sort_values(["group", "abs_elasticity"], ascending=[True, False])
        return d.groupby("group").head(n)

    top_T = topn(sens, "domestic_active", TRANSITIONS, n=3)
    top_P = topn(sens, "domestic_PIs", TRANSITIONS, n=3)
    top_T.to_csv(RESULTS_DIR / "top_interventions_domestic_active.csv", index=False, encoding="utf-8-sig")
    top_P.to_csv(RESULTS_DIR / "top_interventions_domestic_PIs.csv", index=False, encoding="utf-8-sig")

    pnr2 = pnr[pnr["is_within_bounds"] == True].copy()
    pnr2["proximity"] = (pnr2["critical_factor"] - 1).abs()
    closest = pnr2.loc[pnr2.groupby(["group", "target"])["proximity"].idxmin()]
    closest.to_csv(RESULTS_DIR / "closest_point_of_no_return.csv", index=False, encoding="utf-8-sig")

    lines = ["# AI/ML 研究者流動：連立 ODE（ベースライン）\n"]
    lines.append("## 平衡国内活動研究者数 T = D + H_D + P_D と最小存続閾値 M = k × c_bar\n")
    for _, row in summary.iterrows():
        lines.append(
            f"- **{row['group']}**: T_eq = {row['T_equilibrium']:.1f}, "
            f"M = {row['M_threshold']:.1f} (k={int(row['k_used'])}, c_bar={row['c_bar']:.2f}), "
            f"余裕 = {row['margin_to_threshold_T']:.1f}; I = {row['inflow_I']:.4f}\n"
        )

    lines.append("\n## 国内活動研究者数に最も影響を与える上位係数（弾力性）\n")
    for g, df in top_T.groupby("group"):
        lines.append(f"\n### {g}\n")
        for _, r in df.iterrows():
            sign = "+" if r["elasticity"] > 0 else ""
            lines.append(
                f"- {r['rate']}: 弾力性 = {sign}{r['elasticity']:.4f} "
                f"（{r['rate']} を 1% 上げると T は {sign}{r['elasticity']:.2f}% 変化）\n"
            )

    lines.append("\n## 国内 PI 数に最も影響を与える上位係数（弾力性）\n")
    for g, df in top_P.groupby("group"):
        lines.append(f"\n### {g}\n")
        for _, r in df.iterrows():
            sign = "+" if r["elasticity"] > 0 else ""
            lines.append(f"- {r['rate']}: 弾力性 = {sign}{r['elasticity']:.4f}\n")

    lines.append("\n## 各文明圏で最も近い point-of-no-return（T に対する係数変化率）\n")
    for _, r in closest[closest["target"] == "domestic_active"].iterrows():
        action = "増加" if r["direction"] == -1 else ("減少" if r["direction"] == 1 else "影響なし")
        lines.append(
            f"- **{r['group']}**: {r['rate_name']} を現在の {r['critical_factor']:.3f} 倍 "
            f"（{r['current_rate']:.5f} → {r['critical_rate']:.5f}）に{action}させると "
            f"T = M = {r['M_threshold']:.1f} に到達\n"
        )

    with open(RESULTS_DIR / "intervention_summary.md", "w", encoding="utf-8") as f:
        f.writelines(lines)
    print("Wrote summary to", RESULTS_DIR / "intervention_summary.md")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-override", type=int, default=None,
                        help="Override k (number of required independent groups). Default is data-driven.")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    summary_df, sens_df, pnr_df = run_model(k_override=args.k_override, save=not args.no_save)

    print("\n=== Equilibrium domestic active pool T = D + H_D + P_D ===")
    print(summary_df.to_string(index=False))

    print("\n=== Sensitivities (elasticity) ===")
    for (target, g), df in sens_df.groupby(["target", "group"]):
        print(f"\n--- target={target}, group={g} ---")
        print(df[["rate", "value", "elasticity"]].head(8).to_string(index=False))

    print("\n=== Point of no return (rate factor to reach M = k * c_bar) ===")
    print(pnr_df.to_string(index=False))

    if not args.no_save:
        summarize_results()


if __name__ == "__main__":
    main()

