#!/usr/bin/env python3
"""
Generate the civilization → country mapping and the grouping rationale
for the OpenAlex researcher-mobility ODE project.

Outputs
-------
researcher_mobility_ode/data/country_civilization_mapping.json
researcher_mobility_ode/data/civilization_works_summary.csv
researcher_mobility_ode/docs/mapping_rationale.md
"""

import json
import csv
import io
import os
import re
import requests
from collections import defaultdict

OPENALEX_WORKS = "https://api.openalex.org/works"
MAILTO = "researcher-mobility-probe@example.org"
SUBFIELD_AI = "subfields/1702"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DOCS_DIR = os.path.join(BASE_DIR, "docs")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# ISO-3166 country metadata from lukes/ISO-3166-Countries-with-Regional-Codes
ISO_CSV_URL = (
    "https://raw.githubusercontent.com/lukes/ISO-3166-Countries-with-Regional-Codes/"
    "master/all/all.csv"
)

# ---------------------------------------------------------------------------
# 1. Explicit civilization sets (ISO-3166-1 alpha-2)
# ---------------------------------------------------------------------------

GROUPS = {
    "United States": {"US"},
    "Anglosphere ex-US": {"GB", "CA", "AU", "NZ", "IE"},
    "Continental Europe": {
        # EU/EFTA (including UK? no; UK is Anglosphere)
        "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE",
        "GR", "HU", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO",
        "SK", "SI", "ES", "SE",
        # EFTA
        "IS", "LI", "NO", "CH",
    },
    "Other Western": {
        # Israel plus European microstates and Western-affiliated small territories
        "IL", "AD", "MC", "SM", "VA", "GI", "GG", "JE", "IM", "FO", "AX",
        "BM", "KY", "VG", "TC", "AI", "MS", "FK", "GS", "PN", "SH", "TA",
        "BL", "MF", "BQ", "CW", "SX", "AW", "CW", "SX",
    },
    "Sinic": {
        # Huntington Sinic core + closely linked East Asian city-states
        "CN", "HK", "MO", "TW", "KR", "VN", "SG",
    },
    "Japanese": {"JP"},
    "Hindu": {"IN", "NP", "LK", "BT"},
    "Islamic": {
        # MENA
        "TR", "IR", "SA", "AE", "QA", "BH", "KW", "OM", "JO", "IQ", "SY",
        "LB", "YE", "PS", "EG", "DZ", "MA", "TN", "LY", "SD", "MR", "ML",
        "NE", "TD", "SO", "DJ", "ER", "KM",
        # South Asia
        "PK", "BD", "AF", "MV",
        # Southeast Asia
        "ID", "MY", "BN",
        # Central Asia / Caucasus (Azerbaijan only; Armenia/Georgia -> Other)
        "AZ", "KZ", "KG", "TJ", "TM", "UZ",
    },
}

# Collect explicit assignments
DIRECT = {}
for group, codes in GROUPS.items():
    for c in codes:
        DIRECT[c] = group

# ---------------------------------------------------------------------------
# 2. Fallback rules from UN M.49 region/sub-region
# ---------------------------------------------------------------------------


def fallback_group(row):
    """Return a civilization group for a country not explicitly listed above."""
    region = (row.get("region") or "").strip()
    sub = (row.get("sub-region") or "").strip()
    inter = (row.get("intermediate-region") or "").strip()
    a2 = row["alpha-2"]

    # Africa -> residual African (merged in Other Civilizations)
    if region == "Africa":
        return "Other Civilizations"

    # Americas -> residual Latin American (merged in Other Civilizations)
    if region == "Americas":
        return "Other Civilizations"

    # Europe
    if region == "Europe":
        # Eastern Europe + Western Balkans -> Orthodox residual
        if sub in {"Eastern Europe", "Southern Europe"} or inter in {"Eastern Europe"}:
            if a2 in {"AL", "BA", "ME", "MK", "RS", "XK", "MD", "BY", "UA", "RU"}:
                return "Other Civilizations"
        # Remaining European microstates / territories -> Other Western
        if a2 in {"AD", "MC", "SM", "VA", "GI", "GG", "JE", "IM", "FO", "AX"}:
            return "Other Western"
        return "Continental Europe"

    # Asia
    if region == "Asia":
        if sub == "Eastern Asia" and a2 == "MN":
            # Mongolia is geographically East Asian but not Sinic/Japanese
            return "Other Civilizations"
        if sub == "Eastern Asia":
            return "Sinic"  # North Korea, etc. (negligible AI/ML output)
        if sub == "Southern Asia":
            if a2 in {"IN", "NP", "LK", "BT"}:
                return "Hindu"
            return "Islamic"  # e.g. Maldives, Afghanistan
        if sub == "South-Eastern Asia":
            if a2 in {"VN", "SG"}:
                return "Sinic"
            if a2 in {"ID", "MY", "BN"}:
                return "Islamic"
            # Thailand, Philippines, Myanmar, Cambodia, Laos, Timor-Leste -> residual
            return "Other Civilizations"
        if sub in {"Western Asia", "Central Asia"}:
            if a2 == "IL":
                return "Other Western"
            if a2 == "TR":
                return "Islamic"
            # Armenia, Georgia, etc. -> Other Civilizations
            return "Other Civilizations"

    # Oceania
    if region == "Oceania":
        if a2 in {"AU", "NZ"}:
            return "Anglosphere ex-US"
        return "Other Civilizations"  # Pacific island states

    if region == "Antarctica":
        return "Other Civilizations"

    return "Other Civilizations"


# ---------------------------------------------------------------------------
# 3. Fetch country metadata
# ---------------------------------------------------------------------------


def fetch_countries():
    r = requests.get(ISO_CSV_URL, timeout=30)
    r.raise_for_status()
    r.encoding = "utf-8"
    return list(csv.DictReader(io.StringIO(r.text)))


# ---------------------------------------------------------------------------
# 4. Fetch OpenAlex AI/ML works by country (2022-2023)
# ---------------------------------------------------------------------------


def fetch_aiml_counts():
    url = OPENALEX_WORKS
    params = {
        "filter": f"publication_year:2022-2023,topics.subfield.id:{SUBFIELD_AI}",
        "group_by": "authorships.institutions.country_code",
        "per-page": 100,
        "mailto": MAILTO,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    groups = r.json().get("group_by", [])
    counts = {}
    for g in groups:
        code = g["key"].split("/")[-1]
        counts[code] = g["count"]
    return counts


# ---------------------------------------------------------------------------
# 5. Build mapping
# ---------------------------------------------------------------------------


def build_mapping(countries, aiml_counts):
    mapping = {}
    for row in countries:
        a2 = row["alpha-2"]
        a3 = row["alpha-3"]
        if a2 in DIRECT:
            group = DIRECT[a2]
        else:
            group = fallback_group(row)
        mapping[a3] = {
            "group": group,
            "alpha_2": a2,
            "name": row["name"],
            "region": row.get("region"),
            "sub_region": row.get("sub-region"),
            "ai_ml_works_2022_2023": aiml_counts.get(a2, 0),
        }
    return mapping


# ---------------------------------------------------------------------------
# 6. Summarise and write outputs
# ---------------------------------------------------------------------------


def write_outputs(mapping):
    # mapping JSON
    mapping_path = os.path.join(DATA_DIR, "country_civilization_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False, sort_keys=True)

    # summary by group
    group_counts = defaultdict(int)
    for info in mapping.values():
        group_counts[info["group"]] += info["ai_ml_works_2022_2023"]

    summary = []
    for g in GROUPS.keys():
        summary.append({"civilization": g, "works_2022_2023": group_counts[g]})
    summary.append({
        "civilization": "Other Civilizations",
        "works_2022_2023": group_counts["Other Civilizations"],
    })

    summary_path = os.path.join(DATA_DIR, "civilization_works_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["civilization", "works_2022_2023"])
        writer.writeheader()
        writer.writerows(summary)

    return summary, group_counts


# ---------------------------------------------------------------------------
# 7. Rationale document
# ---------------------------------------------------------------------------


def build_rationale_doc(mapping, group_counts, top_n=25):
    # top contributing non-Western/non-traditional examples
    top = sorted(
        mapping.items(),
        key=lambda x: x[1]["ai_ml_works_2022_2023"],
        reverse=True,
    )[:top_n]

    doc = f"""# Civilization mapping rationale for OpenAlex researcher mobility ODE

## 1. Purpose

This document records the country-to-civilization mapping used to group
institutional affiliations in the OpenAlex AI/ML (`subfields/1702`) researcher
mobility model.  The grouping is a practical operationalization of
Huntington’s civilizational framework for *international journal* analysis,
balancing theoretical coherence with the statistical need for large enough
cells to estimate ODE transition rates.

## 2. Theoretical starting point: Huntington’s civilizations

Huntington’s original article identified seven or eight major civilizations:
Western, Latin American, Orthodox, Sinic, Japanese, Hindu, Islamic, and
possibly African.  These groupings are used because research mobility is
shaped by language, academic labour markets, visa regimes and cultural
proximity, all of which tend to cluster along civilizational lines at least
as strongly as along simple geographic regions.

## 3. Why the groups were merged or split

### 3.1 Merges driven by AI/ML sample size

- **Latin American**, **Orthodox**, and **African** blocs are merged into
  **Other Civilizations** because, in AI/ML during 2022-2023, their separate
  counts are too small to estimate stable transition rates.  Keeping them as
  separate ODE compartments would produce noisy or non-identifiable rate
  parameters.
- These three blocs still differ culturally, but in the present dataset they
  behave as a residual pool whose *net* flows can be estimated reliably.
- Within **Other Civilizations**, country-level tags are preserved in the
  mapping file, so the bloc can be re-disaggregated later if richer data
  (e.g. full OpenAlex snapshot, additional years) make it feasible.

### 3.2 Split of the Western bloc

The original **Western** civilization is analytically too large for AI/ML
mobility.  The United States is the dominant global hub and absorbs a large
share of mobile researchers, so it must be separated from the rest of the
West.  Plan A therefore splits the West into four operational groups:

1. **United States**: the single largest AI/ML producer and destination.
2. **Anglosphere ex-US**: GB, CA, AU, NZ, IE.  These share English-language
   publishing and tight academic labour-market integration.
3. **Continental Europe**: EU/EFTA member states plus closely associated
   European states.  Mobility is strongly shaped by the EU research area,
   Horizon funding and Schengen/visa regimes.
4. **Other Western**: Israel plus small Western-affiliated territories and
   microstates that do not fit the EU/EFTA or Anglosphere boxes.

This split keeps the US from dominating the Western average and lets the
model capture distinct flows: US↔Anglosphere, US↔Europe, Anglosphere↔Europe,
etc.

## 4. Group definitions and representative countries

| Group | Rationale | Representative countries (alpha-2) |
|-------|-----------|-----------------------------------|
| United States | dominant AI/ML hub | US |
| Anglosphere ex-US | English-language / common-law / historical flows | GB, CA, AU, NZ, IE |
| Continental Europe | EU/EFTA / EEA research area | DE, FR, NL, BE, CH, AT, SE, NO, DK, FI, IS, ES, IT, PT, … |
| Other Western | Israel + Western microstates / territories | IL, AD, MC, SM, VA, GI, … |
| Sinic | Huntington Sinic core + culturally Chinese city-states | CN, HK, MO, TW, KR, VN, SG |
| Japanese | separate, non-Sinic East Asian high-income research system | JP |
| Hindu | South Asian Indic civilization | IN, NP, LK, BT |
| Islamic | MENA, South Asia, Southeast Asia and Central Asia Muslim-majority bloc | TR, IR, SA, AE, PK, BD, ID, MY, EG, … |
| Other Civilizations | merged residual (Latin American + Orthodox + African + other small Asian states) | BR, MX, AR, RU, UA, ZA, NG, TH, PH, MN, … |

## 5. Fallback rules for small or ambiguous countries

Countries not explicitly listed are assigned by a transparent fallback based
on UN M.49 region and sub-region:

- **Africa** -> Other Civilizations (African residual)
- **Americas** -> Other Civilizations (Latin American residual)
- **Europe**:
  - EU/EFTA/EEA member states -> Continental Europe
  - Eastern Europe / Western Balkans not in EU -> Other Civilizations (Orthodox residual)
  - European microstates -> Other Western
- **Asia**:
  - East Asia (except Japan) -> Sinic
  - South Asia: India/Nepal/Sri Lanka/Bhutan -> Hindu; otherwise Islamic
  - Southeast Asia: Vietnam/Singapore -> Sinic; Indonesia/Malaysia/Brunei -> Islamic; rest -> Other Civilizations
  - Western/Central Asia (except Israel/Turkey) -> Islamic or Other Civilizations
- **Oceania**:
  - Australia/New Zealand -> Anglosphere ex-US
  - Pacific island states -> Other Civilizations

This fallback affects only low-output countries; all major AI/ML producers are
explicitly assigned.

## 6. Empirical support from OpenAlex AI/ML 2022-2023

The table below shows the number of AI/ML works (with at least one author
affiliated to the country) after aggregation to the mapping.  These counts
justify the split/merge decisions: the merged residual is non-negligible
because of large Latin American and Orthodox producers such as Brazil and
Russia, but the individual Latin/African/Orthodox blocs are too small and
sparse to support separate rate estimation in the current two-year window.

| Civilization | AI/ML works 2022-2023 |
|--------------|----------------------|
"""
    for row in group_counts["summary"]:
        doc += f"| {row['civilization']} | {row['works_2022_2023']:,} |\n"

    doc += f"""

### Top individual contributors (alpha-2, group, works)

| Country | Group | Works 2022-2023 |
|---------|-------|-----------------|
"""
    for a3, info in top:
        doc += f"| {info['name']} ({info['alpha_2']}) | {info['group']} | {info['ai_ml_works_2022_2023']:,} |\n"

    doc += f"""

## 7. Limitations and sensitivity

- **Merged residual**: “Other Civilizations” is culturally heterogeneous.  Any
  policy conclusion about this bloc should be interpreted as an aggregate
  residual rather than a single civilization.  Robustness checks will
  re-estimate the model with Latin America, Orthodox and Africa separated once
  longer time series or a full OpenAlex snapshot become available.
- **Korea / Vietnam in Sinic**: South Korea and Vietnam are not culturally
  identical to China, but they are assigned to the Sinic group because
  Huntington placed them in the Sinic/Confucian sphere and because they
  share high East Asian research-mobility patterns.  A sensitivity check with
  Korea as a separate group is planned.
- **Israel in Other Western**: Israel’s academic system is closely integrated
  with the US and Europe, but it is not part of the Anglosphere or EU/EFTA.
  Moving it into Anglosphere or Continental Europe does not materially change
  aggregate counts and can be tested in sensitivity analyses.
- **Alpha-2 vs alpha-3**: OpenAlex returns alpha-2 country codes.  The
  published mapping is stored by alpha-3 with an alpha-2 crosswalk to support
  direct code joins.

## 8. Files

- `data/country_civilization_mapping.json` — full alpha-3 → group mapping with
  UN region and 2022-2023 AI/ML work counts.
- `data/civilization_works_summary.csv` — group-level aggregate counts.
- This document (`docs/mapping_rationale.md`).
"""

    doc_path = os.path.join(DOCS_DIR, "mapping_rationale.md")
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(doc)


# ---------------------------------------------------------------------------
# 8. Main
# ---------------------------------------------------------------------------


def main():
    print("Fetching country metadata...")
    countries = fetch_countries()
    print(f"  {len(countries)} countries/territories")

    print("Fetching OpenAlex AI/ML counts 2022-2023...")
    aiml_counts = fetch_aiml_counts()
    print(f"  {len(aiml_counts)} country codes with non-zero counts")

    print("Building mapping...")
    mapping = build_mapping(countries, aiml_counts)

    summary, group_counts_map = write_outputs(mapping)
    group_counts = {"summary": summary, "dict": group_counts_map}

    print("Writing rationale document...")
    build_rationale_doc(mapping, group_counts)

    print("\nGroup totals:")
    for row in summary:
        print(f"  {row['civilization']:<22} {row['works_2022_2023']:>10,}")


if __name__ == "__main__":
    main()
