# Validation and implementation notes

## Provenance

The source is Daniel's supplied `sigmoid_cohort.py` and `final_aps.py` Colab exports. Public modules retain the scoring functions and APS pipeline, with DataFrame entry points instead of notebook I/O. Player-specific examples, source notebook URLs, and print/display cells were removed. No athlete data is included. Tests and sensitivity probes were added for public reproducibility.

## Safeguards added during packaging

1. RPS sigmoid exponent is clipped to ±700 to avoid numerical overflow at extreme inputs.
2. RPS weighted sums use `min_count=1`: an entirely undefined distribution stays missing instead of becoming a zero score outside the display range.
3. APS raises a descriptive `ValueError` if there are no valid scores, replacing an incidental `KeyError` during aggregation.

These changes do not redesign the normal scoring formulas. The original downloaded files are unchanged.

## Verified behavior

13 unittest checks pass under Python 3.12, pandas 2.2.2, NumPy 1.26.4. They cover bounded and monotonic scores on nondegenerate synthetic input, higher/lower direction, logarithms, one/two/three-attempt weights, aggregation, fallback source, benchmark sample counts, row-order invariance with distinct timestamps, APS band boundaries, and constant-distribution handling.

Run `python validate_stability.py` to reproduce the machine-readable results in `validation/results.json`.

- Synthetic fixture: 40 male and 10 female identifiers in one age group, one drill. Noise probes use one attempt per identifier.
- Multiplicative normal noise (standard deviation 1%, seed 42), with benchmarks recomputed: mean absolute RPS change 0.450, maximum 0.937; mean absolute APS change 0.452, maximum 1.800.
- Replacing the highest male score with 1e9 left other players' RPS unchanged in this fixture. This tests one upper-tail outlier, not general contamination robustness.
- Hard-fallback stress probe: 29 male observations, 10 female observations shifted upward by 100 score units; adding one male observation changes the benchmark source. Maximum existing-player RPS change: 23.825. This deliberately adversarial example demonstrates a real discontinuity, not typical observed impact.
- Constant scores: 50 missing RPS attempt and player scores; APS returns an explicit validation error. There is insufficient variation to calibrate either reference.

## Unresolved methodological limitations

- **APS lower-is-better anchors:** negating and reversing q30/q50/q70/q90 does not preserve their percentile labels. With raw anchors 30/50/70/90, raw score 50 maps to APS 70 for lower-is-better. Scores remain monotonic, but median calibration differs. Define anchors on oriented values before estimating quantiles if percentile semantics are intended; this would be a methodology change and is not silently applied here.
- **Hard cohort threshold:** RPS can jump at 30 observations. Consider calibrated pooling or continuous blending, followed by sensitivity tests with representative private data.
- **Different benchmark grain:** RPS uses all attempts, so frequent participants affect benchmarks more; APS uses one latest observation per player/assessment.
- **No frozen APS reference:** recomputation changes anchors and may move bands without athlete improvement. Version benchmarks before claiming stable longitudinal bands.
- **Unequal coverage:** overall scores average available assessments/categories. Athletes assessed on different drills may not be directly comparable. No minimum coverage policy is enforced.
- **Input quality:** mappings, identity, duplicates, non-finite values, invalid log inputs, and equal timestamps need explicit production policies. Defaults do not validate a real dataset. Partially missing attempt scores are not automatically renormalized in RPS.
- **No outcome validation:** these checks establish selected software properties, not athletic validity, fairness across populations, predictive accuracy, or production readiness.

Pandas emits a future-compatibility warning for `groupby.apply`; dependencies are pinned to the tested versions. Migration to a new pandas version should rerun the suite and compare outputs.
