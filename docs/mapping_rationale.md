# Civilization mapping rationale for OpenAlex researcher mobility ODE

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

The table below shows the number of AI/ML works with at least one author
affiliated to the country in the mapping group.  Because one work can list
authors from multiple countries, group totals are *country-work incidences*
and will exceed the number of unique works (919,060 in this window).  These
incidences justify the split/merge decisions: the merged residual is
non-negligible because of large Latin American and Orthodox producers such as
Brazil and Russia, but the *author-level cross-bloc transitions* (not raw work
counts) are still too sparse to estimate separate ODE rate matrices for Latin
American, Orthodox and African blocs in the current two-year window.

| Civilization | AI/ML works 2022-2023 |
|--------------|----------------------|
| United States | 100,154 |
| Anglosphere ex-US | 66,221 |
| Continental Europe | 156,002 |
| Other Western | 3,847 |
| Sinic | 175,918 |
| Japanese | 16,929 |
| Hindu | 53,745 |
| Islamic | 114,388 |
| Other Civilizations | 46,466 |


### Top individual contributors (alpha-2, group, works)

| Country | Group | Works 2022-2023 |
|---------|-------|-----------------|
| China (CN) | Sinic | 140,788 |
| United States of America (US) | United States | 100,154 |
| India (IN) | Hindu | 52,279 |
| Indonesia (ID) | Islamic | 43,436 |
| United Kingdom of Great Britain and Northern Ireland (GB) | Anglosphere ex-US | 29,820 |
| Germany (DE) | Continental Europe | 29,130 |
| France (FR) | Continental Europe | 23,071 |
| Italy (IT) | Continental Europe | 18,284 |
| Canada (CA) | Anglosphere ex-US | 17,105 |
| Japan (JP) | Japanese | 16,929 |
| Australia (AU) | Anglosphere ex-US | 14,856 |
| Spain (ES) | Continental Europe | 13,125 |
| Korea, Republic of (KR) | Sinic | 11,953 |
| Russian Federation (RU) | Other Civilizations | 11,489 |
| Netherlands, Kingdom of the (NL) | Continental Europe | 9,509 |
| Brazil (BR) | Other Civilizations | 9,323 |
| Saudi Arabia (SA) | Islamic | 8,981 |
| Türkiye (TR) | Islamic | 8,823 |
| Switzerland (CH) | Continental Europe | 7,625 |
| Malaysia (MY) | Islamic | 7,375 |
| Iran, Islamic Republic of (IR) | Islamic | 6,782 |
| Hong Kong (HK) | Sinic | 6,763 |
| Singapore (SG) | Sinic | 6,329 |
| Pakistan (PK) | Islamic | 5,850 |
| Poland (PL) | Continental Europe | 5,653 |


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

## 9. References

1. Huntington, S. P. (1993). The Clash of Civilizations? *Foreign Affairs*,
   72(3), 22–49. https://doi.org/10.2307/20045621
