# AI/ML researcher mobility ODE pilot

This repository builds a reproducible OpenAlex pipeline for an AI/ML (Computer Science subfield `1702`) researcher-cohort, estimates per-civilisation transition rates, and runs a coupled linear ODE model to identify intervention priorities and point-of-no-return thresholds.

## Pipeline

1. **Data extraction** (`src/openalex_client.py`)
   - Queries `https://api.openalex.org/works` with subfield `subfields/1702` and per-civilisation country filters.
   - Uses cursor pagination, on-disk cache in `data/cache/`, and batched author lookups.
   - `mailto=researcher-mobility-probe@example.org` is used for the polite pool.

2. **Cohort & rate estimation** (`src/cohort_extraction.py`)
   - Defines an AI/ML cohort with career start 2000–2016, `>=10` AI/ML works.
   - Reproduces the note's 2×2 trajectory logic (`D`, `A`, `H`, `P`) and adds `PI` (`last-author` on `>=6` author paper) and `dropout` (no publication after `DROPOUT_LATEST_YEAR`).
   - Stratified sampling by civilisation group.
   - Estimates hazard rates with Laplace smoothing and a rate cap.

3. **ODE model** (`src/ode_model.py`)
   - Six-state compartment model per civilisation: `D`, `A`, `H_D`, `H_A`, `P_D`, `P_A`.
   - Constant annual inflow `I` from observed new-entrant rates.
   - Equilibrium, transition-rate elasticities, and point-of-no-return scans against `M = k * c_bar`.

4. **Endogenous ODE model** (`src/ode_model_endogenous.py`)
   - Adds PI-driven domestic inflow `I(P_D) = I0 + r * P_D`.
   - `r` is capped at half the critical reproduction rate (`r_critical`) that keeps the linear 6x6 system stable (all eigenvalues negative).
   - Computes equilibrium, sensitivities for `domestic_active` (`T`) and `domestic_PIs` (`P`), and point-of-no-return thresholds with monotonic-direction scans.

## Reproduce

```bash
# Requires Python 3.10+ and packages in src/requirements.txt (pandas, numpy, scipy, requests)
python src/openalex_client.py
python src/cohort_extraction.py
python src/ode_model.py
python src/ode_model_endogenous.py
```

`data/cache/` is excluded from git; the API will re-populate it on first run.

## Outputs

- `data/cohort/cohort.csv` — classified cohort (273 authors in this pilot)
- `data/cohort/transition_rates.csv` — per-civilisation rates (`alpha`, `beta`, `h_D`, `h_A`, `p_D`, `p_A`, `d`)
- `data/cohort/raw_sampled_works.json` — sampled works used for coauthor statistics
- `results/equilibrium_summary.csv` — baseline equilibrium `T` vs `M`
- `results/sensitivity.csv` — baseline elasticities of `T` and `P` to each transition rate
- `results/point_of_no_return.csv` — baseline critical multipliers at which `T` reaches `M`
- `results/intervention_summary.md` — human-readable baseline top interventions per civilisation
- `results/endogenous/equilibrium_summary.csv` — endogenous inflow equilibrium
- `results/endogenous/sensitivity.csv` — endogenous elasticities
- `results/endogenous/point_of_no_return.csv` — endogenous critical multipliers
- `results/endogenous/top_inflow_T.csv`, `top_transitions_T.csv`, etc. — ranked intervention levers
- `results/endogenous/intervention_summary.md` — human-readable endogenous summary

## Limitations

- The PI-driven inflow is linear and `r` is capped at half the critical reproduction rate to keep the linear system stable. A saturating recruitment function would be required for a fully structural calibration.
- `r_obs` (observed cross-sectional new-entrant / PI ratio) exceeds the stability cap for every civilisation; the reported `r` is therefore a conservative, stable value and the true feedback effect of `p_D` and `h_D` on `T` is likely larger.
- Dropout is treated as a single departure hazard; future versions will use Aalen-Johansen competing-events rates as in the original note.
- The cohort sample is small for some groups (`Other Western`, `Japanese`); confidence intervals are not yet computed.

## Sources

- OpenAlex API: <https://api.openalex.org>
- Subfield: `subfields/1702` (Artificial Intelligence) under `fields/17` (Computer Science)
- Civilisation grouping rationale: `docs/mapping_rationale.md`
