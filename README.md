# Athlete Scoring — Sogility

> How can a training score explain peer context and recent performance?

**Industry practicum · Python reference implementation · synthetic validation**

[Portfolio case study](https://daniel-kang-vs.github.io/projects/sogility.html) · [Validation and limitations](VALIDATION.md)

## The problem

Global comparisons can obscure developmental differences between athletes. Repeated attempts are noisy, and small cohorts can produce unstable benchmarks. This project separates a cohort-relative score (RPS) from an assessment-based performance band (APS), supporting distinct parent and trainer views.

## Public evidence

This repository now includes importable extracts from Daniel's supplied `sigmoid_cohort.py` and `final_aps.py` notebook exports. Personal examples, notebook links, confidential input data, and internal interfaces are excluded. The original files remain unchanged locally. These are analytical reference implementations, not evidence of a production deployment.

## Scoring logic

### RPS: relative performance

- Log-transform selected drills and orient results so higher means better.
- Fit median and 75th-percentile span to the age-and-gender cohort when it has at least 30 **attempt observations**. Otherwise use the same age group across genders; invalid parameters fall back to global benchmarks.
- Map oriented results through `30 + 69 / (1 + exp(-z))`, with `z = 1.0986 * (x - median) / abs(span - median)`.
- Combine the newest three attempts with 0.50 / 0.30 / 0.20 weights. Two attempts use 2/3 and 1/3; one uses 100%.
- Average assessment scores for the overall player score. A score of 75 is **not** an empirical 75th percentile. The benchmark median maps to 64.5.

```python
from sigmoid_cohort import compare_global_vs_cohort_sigmoid_rps

attempts, global_params, cohort_params, age_params, diagnostics = (
    compare_global_vs_cohort_sigmoid_rps(
        prepared_attempts,
        skill_type_map={"Freelap 10yd Dash": "Log"},
        good_direction_map={"Freelap 10yd Dash": "Lo"},
    )
)
player_scores = diagnostics["overall_player_rps"]
```

Supply all applicable drill mappings explicitly: unmapped RPS drills default to normal / higher-is-better. The RPS function takes raw scores and logs inside the function.

### APS: performance bands

- Build benchmarks from each player's **latest attempt per assessment**, preventing repeat attempts from increasing that player's benchmark weight.
- Use age-and-gender cohorts with at least 30 players per assessment; otherwise combine genders within the age group. This implementation has **no global APS fallback**.
- Interpolate through q30 / q50 / q70 / q90 anchors to 30 / 50 / 70 / 90; clip APS to **0–100**.
- Weight recent attempts using the same recency rules as RPS, average assessments within each skill category, then equally average available categories.
- Assign Foundation (<30), Developing (30–<50), Approaching (50–<70), Strong (70–<90), or Elite (≥90).

```python
from final_aps import run_aps

result = run_aps(prepared_aps_attempts)
player_bands = result["players"]
benchmark_audit = result["small_samples"]
```

APS input scores must already be log-transformed for the selected log-type drills, as in the notebook preprocessing. Include the boolean `higher_is_better` column. Do not log the same input twice. Category mapping excludes unmapped drills and returns an audit table.

## Input contract

Both entry points accept a pandas DataFrame with `category_assessment`, `score`, `timestamp`, `gender_clean`, `age_group`, and `cohort`. Default identity is `first_name` + `last_name` + gender + age group; public examples use synthetic identifiers. RPS supports `player_id_col`; APS uses `athlete_id` when provided. APS also requires `higher_is_better`. Clean invalid timestamps, non-finite scores, and nonpositive log inputs before calling APS. Use stable player IDs for any real integration.

## Run and inspect

Python 3.12 was used for local verification. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python validate_stability.py
```

The examples in the tests generate synthetic data in memory. No confidential CSV is needed. The validation script writes `validation/results.json` deterministically with seed 42.

## Results and interpretation

**13 functional checks pass**, including score bounds, monotonicity, recency weights, benchmark grain, fallback selection, input-row ordering, and degenerate distributions. In one synthetic 1% noise experiment, mean absolute changes were **0.45 RPS points** and **0.45 APS points**. This is a limited sensitivity experiment, not a general stability guarantee.

A deliberately separated age/gender stress case produced a **23.82-point RPS jump** when a cohort crossed from 29 to 30 observations. Lower-is-better APS anchors also require review. See [VALIDATION.md](VALIDATION.md) for exact limitations and safeguards added during packaging.

## Report versus executable implementation

The final report identifies these two filenames, but some narrative details differ from their code:

- RPS uses hard benchmark fallback, not gradual cohort/global blending.
- APS uses q30/q50/q70/q90 and 0–100, not q10/q25/q50/q75/q90 with a 30–99 rescale.
- Two-attempt weights are 2/3 and 1/3, not renormalized 0.5 and 0.3.
- APS benchmarks are recalculated on each run, so bands are not fixed over time. Frozen, versioned benchmark tables are a next step.

This README describes the executable implementation. Neither these tests nor the report establishes prediction of match performance; outcome labels were unavailable.

## Repository guide

- `sigmoid_cohort.py` — RPS scoring, aggregation, and diagnostics.
- `final_aps.py` — APS scoring, category aggregation, and audit tables.
- `tests/test_scoring.py` — synthetic functional tests.
- `validate_stability.py` — reproducible sensitivity and edge-case probes.
- `validation/results.json` — recorded probe results.
- `VALIDATION.md` — limitations, provenance, and packaging changes.

## Product and handoff

The broader practicum explored parent views for score, skills and history, and trainer views for entry and comparison. Python holds the analytical reference; the React/TypeScript prototype and proposed Elixir production handoff are separate artifacts and are not included here. SQL supported the original extraction work. This repository does not claim that the scoring system was deployed.
