# Reproducibility test plan: researcher_mobility_ode extensions (PR #330)

## Scope
Run the cached-data ODE pipeline and all new extension scripts on the current checkout of `researcher_mobility_ode` and verify that every script exits cleanly, writes the expected CSV/MD artifacts, and passes sanity checks on the numerical outputs. No live OpenAlex API calls are needed.

## Assumptions
- Repository root: `/home/ubuntu/repos/wip`
- Working directory for commands: `/home/ubuntu/repos/wip/researcher_mobility_ode`
- Python 3.10+ with `numpy`, `scipy`, `pandas` installed.
- Cached cohort files present:
  - `data/cohort/cohort.csv`
  - `data/cohort/transition_rates.csv`
  - `data/cohort/raw_sampled_works.json`
  - `data/country_civilization_mapping.json`

## Steps and pass/fail criteria

### 0. Environment sanity
Command: `python -c "import sys; import numpy, scipy, pandas; print(sys.version); ..."`
- **Pass**: Python >= 3.10, numpy/scipy/pandas available.

### 1. Endogenous ODE (default)
Command: `python src/ode_model_endogenous.py > test_outputs/endogenous_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/endogenous/intervention_summary.md` exists and is non-empty.
- **Pass**: `results/endogenous/equilibrium_summary.csv` has 9 rows, all `T_equilibrium > M_threshold` (`margin_to_threshold_T > 0`).
- **Spot-check**: in `results/endogenous/sensitivity.csv`, for `group='Japanese'`, `target='domestic_active'`:
  - `p_D` elasticity is positive and between `0.20` and `0.25`.
  - `d` elasticity is negative and between `-2.5` and `-1.8`.

### 2. Endogenous ODE with saturating inflow
Command: `python src/ode_model_endogenous.py --saturating --results-dir results/endogenous_saturating > test_outputs/endogenous_saturating_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/endogenous_saturating/equilibrium_summary.csv` exists and is non-empty.
- **Pass**: the CSV contains an `epsilon` column with positive values for groups with `P_D_obs > 0`.
- **Pass**: all rows have `T_equilibrium > M_threshold` and finite values.

### 3. k-sensitivity
Command: `python src/k_sensitivity.py > test_outputs/k_sensitivity_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/k_sensitivity/k_sensitivity.csv` exists, has `9 * 6 = 54` rows (9 groups x 6 multipliers).
- **Pass**: for `k_multiplier == 1.0`, all groups have `T_equilibrium > M_threshold`.

### 4. Bootstrap CI
Command: `python src/bootstrap_ci.py > test_outputs/bootstrap_ci_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/bootstrap_ci/bootstrap_draws.csv` and `results/bootstrap_ci/bootstrap_summary.csv` exist and are non-empty.
- **Pass**: `bootstrap_summary.csv` has 9 rows, `n == 200` for each group, no NaNs in `T_equilibrium_mean` or `margin_mean`.

### 5. Time-varying parameters
Command: `python src/time_varying.py > test_outputs/time_varying_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/time_varying/` contains `equilibrium_summary.csv`, `sensitivity.csv`, `point_of_no_return.csv`, `period_comparison.csv`.
- **Pass**: `equilibrium_summary.csv` has rows for both `early` and `late` periods and `T_equilibrium > M_threshold` for all rows.
- **Pass**: `period_comparison.csv` has at least 4 rows (one per group with both periods) and numeric `delta_T` values.

### 6. Country-level resolution
Command: `PYTHONHASHSEED=0 python src/country_level.py > test_outputs/country_level_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/country_level/` contains `equilibrium_summary.csv`, `sensitivity.csv`, `point_of_no_return.csv`.
- **Pass**: `equilibrium_summary.csv` has at least 1 row, all rows have `T_equilibrium > M_threshold`.
- **Note**: without `PYTHONHASHSEED=0` the script can produce slightly different rows on each run because `author_works_map` and `_most_common_country` iterate over Python `set` objects whose order is hash-randomized.  Use `PYTHONHASHSEED=0` (or fix the source to sort country codes) for deterministic output.

### 7. Network externalities
Command: `python src/network_externalities.py > test_outputs/network_externalities_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/network_externalities/network_eq_spillover_0.1.csv` and `collaboration_matrix.csv` exist and are non-empty.
- **Pass**: `network_eq_spillover_0.1.csv` has 9 rows, `max_eigenvalue_real < 0`, all `T_equilibrium > M_threshold`.

### 8. Behavioral layers
Command: `python src/behavioral_layers.py > test_outputs/behavioral_layers_run.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/behavioral_layers/behavioral_layers_counterfactual.csv` exists and is non-empty.
- **Pass**: file has numeric `delta_T` and `delta_margin` columns.

### 9. Policy counterfactuals
Command: `python src/policy_counterfactuals.py --packages > test_outputs/policy_counterfactuals.log 2>&1`
- **Pass**: exit code 0.
- **Pass**: `results/policy_counterfactuals/counterfactuals.csv` (324 data rows plus header) and `ranked_interventions.csv` exist and are non-empty.
- **Spot-check**: in `ranked_interventions.csv`, the top-ranked intervention per group (`groupby('group').head(1)`) has `lever == 'd'` and `direction == 'decrease'` for all groups, reflecting the dominance of dropout reduction.
- **Note**: without `--packages` the script only generates single-lever counterfactuals.  Use `--packages` to include the multi-lever policy packages present in the committed reference output.

## Artifacts to collect
- `test_outputs/*.log` for each script.
- Any generated CSV/MD under `results/endogenous/`, `results/endogenous_saturating/`, `results/k_sensitivity/`, `results/bootstrap_ci/`, `results/time_varying/`, `results/country_level/`, `results/network_externalities/`, `results/behavioral_layers/`, `results/policy_counterfactuals/`.
- A final `test_outputs/verification_report.txt` summarising pass/fail per step.
