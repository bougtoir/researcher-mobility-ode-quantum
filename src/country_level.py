#!/usr/bin/env python3
"""
Country-level ODE resolution for major AI/ML research nations.

This script derives each cohort author's most likely country from the
stratified sample works, then re-runs the endogenous ODE for a selected set
of major countries.  It is intended as a finer-grained complement to the
civilisation-level model: it shows, for example, how vulnerable the United
States, China, Japan, or Germany are independent of their broader group.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import ode_model_endogenous as odm
from cohort_extraction import estimate_rates, ABROAD_WINDOW


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results" / "country_level"
CACHE_FILE = BASE_DIR / "data" / "cohort" / "raw_sampled_works.json"

DEFAULT_COUNTRY_CODES = [
    "US", "CN", "GB", "DE", "JP", "IN", "KR", "CA", "FR", "AU",
    "IL", "SG", "IT", "NL", "CH", "ES", "SE", "BR", "RU", "TW",
]


def _author_id(raw_id):
    return raw_id.split("/")[-1] if raw_id else None


def build_country_mapping(codes):
    """Return alpha-2 -> country-name mapping restricted to requested codes."""
    with open(BASE_DIR / "data" / "country_civilization_mapping.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    a2g = {}
    name_lookup = {}
    for a3, info in mapping.items():
        a2 = info.get("alpha_2")
        name = info.get("name")
        if a2 and name and a2 in codes:
            a2g[a2] = name
            name_lookup[name] = a2
    return a2g, name_lookup


def author_works_map(sample_path, requested_codes):
    """Build aid -> list of {year, countries} from the sampled works cache.

    Only country codes present in requested_codes are retained so that origin,
    abroad, and recent labels are all consistent with the country-level model.
    """
    with open(sample_path, "r", encoding="utf-8") as f:
        works = json.load(f)
    out = defaultdict(list)
    for w in works:
        y = w.get("publication_year")
        for auth in w.get("authorships", []):
            aid = _author_id((auth.get("author") or {}).get("id"))
            if not aid:
                continue
            countries = set()
            for inst in auth.get("institutions", []):
                cc = inst.get("country_code")
                if cc and cc in requested_codes:
                    countries.add(cc)
            if countries:
                out[aid].append({"year": y, "countries": countries})
    return out


def _most_common_country(works, a2g):
    counter = Counter()
    for w in works:
        for cc in sorted(w.get("countries", set())):
            counter[a2g[cc]] += 1
    return counter.most_common(1)[0][0] if counter else None


def origin_country(works, career_start, a2g):
    """Most common country in the first three years of the career window."""
    first_window = [
        w for w in works
        if w["year"] is not None and w["year"] <= career_start + 2
    ]
    g = _most_common_country(first_window, a2g)
    if g is None:
        # Fall back to the most common country seen anywhere in the sample.
        g = _most_common_country(works, a2g)
    return g


def abroad_country(works, origin, career_start, a2g):
    """True if any sampled work within ABROAD_WINDOW has an institution outside the origin country."""
    for w in sorted(works, key=lambda x: x["year"] if x["year"] is not None else 9999):
        y = w["year"]
        if y is None or y > career_start + ABROAD_WINDOW:
            break
        for cc in w["countries"]:
            if a2g.get(cc) and a2g[cc] != origin:
                return True
    return False


def recent_country(works, a2g):
    """Most common country in sampled works from 2021 onward."""
    recent = [w for w in works if w["year"] is not None and w["year"] >= 2021]
    return _most_common_country(recent, a2g)


def relabel_cohort(cohort, sample_path, requested_codes, min_cohort, a2g):
    """Rebuild origin_group, abroad, and recent_group at country resolution."""
    works_map = author_works_map(sample_path, requested_codes)

    rows = []
    for _, row in cohort.iterrows():
        aid = row["author_id"]
        works = works_map.get(aid, [])
        if not works:
            continue
        origin = origin_country(works, row["career_start"], a2g)
        if origin is None:
            continue
        abroad = abroad_country(works, origin, row["career_start"], a2g)
        recent = recent_country(works, a2g)

        row = row.copy()
        row["origin_country"] = origin
        row["abroad"] = abroad
        row["recent_country"] = recent
        rows.append(row)

    cohort = pd.DataFrame(rows)
    valid_names = set(a2g.values())
    cohort = cohort[cohort["origin_country"].isin(valid_names)]
    vc = cohort["origin_country"].value_counts()
    cohort = cohort[cohort["origin_country"].map(vc) >= min_cohort]

    cohort["origin_group"] = cohort["origin_country"]
    cohort["recent_group"] = cohort["recent_country"]
    return cohort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--country-codes", nargs="+", default=DEFAULT_COUNTRY_CODES,
                        help="ISO-3166-1 alpha-2 codes to model at country level.")
    parser.add_argument("--safety-factor", type=float, default=0.5)
    parser.add_argument("--min-cohort", type=int, default=10,
                        help="Minimum cohort size to include a country.")
    args = parser.parse_args()

    a2g, _ = build_country_mapping(args.country_codes)
    cohort = pd.read_csv(odm.COHORT_DIR / "cohort.csv")
    cohort = relabel_cohort(
        cohort, CACHE_FILE, set(a2g.keys()), args.min_cohort, a2g
    )

    if len(cohort) == 0:
        raise ValueError("No cohort authors matched the requested country codes")

    rates = estimate_rates(cohort).set_index("group")
    summary, sens, pnr = odm.run_endogenous_model(
        save=False,
        safety_factor=args.safety_factor,
        cohort_df=cohort,
        rates_df=rates,
        a2g=a2g,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(RESULTS_DIR / "equilibrium_summary.csv", index=False, encoding="utf-8-sig")
    sens.to_csv(RESULTS_DIR / "sensitivity.csv", index=False, encoding="utf-8-sig")
    pnr.to_csv(RESULTS_DIR / "point_of_no_return.csv", index=False, encoding="utf-8-sig")

    print("=== Country-level equilibrium ===")
    print(summary[["group", "T_equilibrium", "M_threshold", "margin_to_threshold_T",
                    "I0", "r", "r_obs", "r_critical"]].to_string(index=False))
    print(f"\nSaved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
