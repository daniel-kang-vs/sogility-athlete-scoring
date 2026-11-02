"""APS reference extracted from the supplied notebook export; see VALIDATION.md for documented edge-case safeguards."""
import numpy as np
import pandas as pd

def map_assessment_category(assessment):
    """
    Maps each assessment into one of five APS parent categories:
    Dribbling, Passing, First Touch, Agility, Vision.
    Unmapped assessments are excluded from scoring later.
    """
    if pd.isna(assessment):
        return "Unmapped"

    a = str(assessment).lower()

    if "circuit" in a:
        return "Dribbling"

    elif "icon" in a:
        return "Passing"

    elif (
        "tech touch" in a
        or "techtouch" in a
        or "1st touch" in a
        or "one touch" in a
    ):
        return "First Touch"

    elif (
        "5-10-5" in a
        or "shuttle" in a
        or "sprint" in a
        or "10yd" in a
        or "10 yd" in a
        or "10yrd" in a
        or "10 yrd" in a
        or "dash" in a
    ):
        return "Agility"

    elif (
        "go 360" in a
        or "scan" in a
        or "react" in a
        or "reaction" in a
        or "large to small" in a
        or "small to large" in a
        or "balance" in a
        or "vision wall" in a
        or "roxpro" in a
        or "reflection edge" in a
    ):
        return "Vision"

    return "Unmapped"

def interpolate_score(x, x0, x1, y0, y1):
    if pd.isna(x0) or pd.isna(x1) or x1 == x0:
        return np.nan

    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)

def aps_from_benchmarks(score, q30, q50, q70, q90, higher_is_better=True):
    """
    Converts raw score into APS using benchmark anchors:
    q30 -> 30
    q50 -> 50
    q70 -> 70
    q90 -> 90

    Final APS is clipped between 0 and 100.
    """
    if any(pd.isna(v) for v in [score, q30, q50, q70, q90]):
        return np.nan

    if not higher_is_better:
        score = -score
        q30, q50, q70, q90 = sorted([-q90, -q70, -q50, -q30])

    if score <= q30:
        aps = interpolate_score(score, q30 - (q50 - q30), q30, 0, 30)

    elif score <= q50:
        aps = interpolate_score(score, q30, q50, 30, 50)

    elif score <= q70:
        aps = interpolate_score(score, q50, q70, 50, 70)

    elif score <= q90:
        aps = interpolate_score(score, q70, q90, 70, 90)

    else:
        aps = interpolate_score(score, q90, q90 + (q90 - q70), 90, 100)

    return float(np.clip(aps, 0, 100))

def assign_aps_band(score):
    """
    Converts APS score into performance band.
    """
    if pd.isna(score):
        return np.nan
    elif score < 30:
        return "Foundation"
    elif score < 50:
        return "Developing"
    elif score < 70:
        return "Approaching"
    elif score < 90:
        return "Strong"
    else:
        return "Elite"

def get_recent_weights(n):
    """
    Recent attempt weighting rule:
    1 attempt: 100%
    2 attempts: most recent 2/3, second most recent 1/3
    3+ attempts: most recent 0.5, second 0.3, third 0.2
    """
    if n == 1:
        return [1.0]
    elif n == 2:
        return [2 / 3, 1 / 3]
    else:
        return [0.5, 0.3, 0.2]

def weighted_recent_assessment_score(group):
    """
    Calculates weighted APS for one player and one assessment.
    Uses only the most recent 1, 2, or 3 attempts.
    """
    group = group.sort_values("timestamp", ascending=False).head(3).copy()

    weights = get_recent_weights(len(group))
    group["recent_weight"] = weights

    weighted_score = np.average(group["APS"], weights=group["recent_weight"])

    return pd.Series({
        "weighted_assessment_aps": weighted_score,
        "individual_assessment_band": assign_aps_band(weighted_score),
        "n_recent_scores_used": len(group),
        "latest_assessment_date": group["timestamp"].max(),
        "weights_used": ", ".join([str(round(w, 3)) for w in weights]),
        "raw_attempt_aps_used": ", ".join([str(round(v, 1)) for v in group["APS"]])
    })

def run_aps(df, min_cohort_n=30):
    """Accept prepared attempts; score is already logged for log-type drills.

    Required columns: first_name, last_name (or athlete_id), gender_clean,
    age_group, cohort, category_assessment, score, timestamp, higher_is_better.
    Benchmarks are recomputed on every call; they are not frozen references.
    """
    MIN_COHORT_N = min_cohort_n
    player_cols = (["athlete_id"] if "athlete_id" in df.columns else
                   ["first_name", "last_name", "gender_clean", "age_group"])
    df = df.copy()

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["aps_category"] = df["category_assessment"].apply(map_assessment_category)

    # Keep unmapped rows in df for audit, but exclude them from all scoring calculations.
    scoring_input = (
        df
        .dropna(subset=["score", "timestamp", "category_assessment", "cohort", "age_group"])
        .query("aps_category != 'Unmapped'")
        .copy()
    )


    # ==========================================================
    # 3. BUILD BENCHMARK BASE
    # ==========================================================
    # Benchmark base uses each player's most recent attempt per assessment.
    # Unmapped assessments are already excluded.

    benchmark_base = (
        scoring_input
        .sort_values("timestamp")
        .drop_duplicates(subset=player_cols + ["category_assessment"], keep="last")
        .copy()
    )


    # ==========================================================
    # 4. COHORT SAMPLE SIZE CHECK
    # ==========================================================

    cohort_sizes = (
        benchmark_base
        .groupby(["category_assessment", "cohort"])
        .size()
        .reset_index(name="n_original_cohort")
    )

    benchmark_base = benchmark_base.merge(
        cohort_sizes,
        on=["category_assessment", "cohort"],
        how="left"
    )

    benchmark_base["combined_for_small_sample"] = (
        benchmark_base["n_original_cohort"] < MIN_COHORT_N
    )

    benchmark_base["benchmark_type"] = np.where(
        benchmark_base["combined_for_small_sample"],
        "age_group",
        "age_gender"
    )

    benchmark_base["aps_distribution_used"] = np.where(
        benchmark_base["combined_for_small_sample"],
        "Combined with same age group across gender",
        "Original age-gender cohort"
    )

    benchmark_base["benchmark_cohort"] = np.where(
        benchmark_base["benchmark_type"] == "age_group",
        benchmark_base["age_group"].astype(str),
        benchmark_base["cohort"]
    )


    # ==========================================================
    # 5. BUILD BENCHMARK SOURCE
    # ==========================================================
    # This guarantees that age-group fallback truly combines both genders.

    benchmark_source_age_gender = benchmark_base.copy()
    benchmark_source_age_gender["benchmark_cohort"] = benchmark_source_age_gender["cohort"]

    benchmark_source_age = benchmark_base.copy()
    benchmark_source_age["benchmark_cohort"] = benchmark_source_age["age_group"].astype(str)

    benchmark_source = pd.concat(
        [benchmark_source_age_gender, benchmark_source_age],
        ignore_index=True
    )


    # ==========================================================
    # 6. BUILD BENCHMARK TABLE
    # ==========================================================

    benchmark_quantiles = [0.30, 0.50, 0.70, 0.90]

    benchmark_table = (
        benchmark_source
        .groupby(["category_assessment", "benchmark_cohort"])["score"]
        .quantile(benchmark_quantiles)
        .unstack()
        .reset_index()
    )

    benchmark_table.columns = [
        "category_assessment",
        "benchmark_cohort",
        "q30",
        "q50",
        "q70",
        "q90"
    ]

    benchmark_sample_sizes = (
        benchmark_source
        .groupby(["category_assessment", "benchmark_cohort"])
        .size()
        .reset_index(name="n_benchmark_distribution")
    )

    benchmark_table = benchmark_table.merge(
        benchmark_sample_sizes,
        on=["category_assessment", "benchmark_cohort"],
        how="left"
    )


    # ==========================================================
    # 7. MERGE BENCHMARKS BACK TO ALL VALID ATTEMPTS
    # ==========================================================

    benchmark_lookup_cols = (
        player_cols
        + [
            "category_assessment",
            "n_original_cohort",
            "combined_for_small_sample",
            "benchmark_type",
            "aps_distribution_used",
            "benchmark_cohort"
        ]
    )

    benchmark_lookup = benchmark_base[benchmark_lookup_cols].drop_duplicates()

    scoring_df = scoring_input.merge(
        benchmark_lookup,
        on=player_cols + ["category_assessment"],
        how="left"
    )

    scoring_df = scoring_df.merge(
        benchmark_table,
        on=["category_assessment", "benchmark_cohort"],
        how="left"
    )


    # ==========================================================
    # 8. CALCULATE APS FOR EACH RAW ATTEMPT
    # ==========================================================

    scoring_df["APS"] = scoring_df.apply(
        lambda row: aps_from_benchmarks(
            score=row["score"],
            q30=row["q30"],
            q50=row["q50"],
            q70=row["q70"],
            q90=row["q90"],
            higher_is_better=row["higher_is_better"]
        ),
        axis=1
    )

    scoring_df["APS_band_raw_attempt"] = scoring_df["APS"].apply(assign_aps_band)


    # ==========================================================
    # 9. CALCULATE WEIGHTED INDIVIDUAL ASSESSMENT SCORES
    # ==========================================================
    # Unmapped assessments are not present in scoring_df.

    valid_scores = (
        scoring_df
        .dropna(subset=["APS", "aps_category", "timestamp"])
        .query("aps_category != 'Unmapped'")
        .copy()
    )

    if valid_scores.empty:
        raise ValueError("No valid APS scores: check empty input, unmapped drills, or degenerate benchmarks.")

    assessment_recent_scores = (
        valid_scores
        .groupby(player_cols + ["aps_category", "category_assessment"], as_index=False)
        .apply(weighted_recent_assessment_score)
        .reset_index(drop=True)
    )

    assessment_recent_scores["weighted_assessment_aps"] = (
        assessment_recent_scores["weighted_assessment_aps"].round(1)
    )


    # ==========================================================
    # 10. CALCULATE CATEGORY SCORES
    # ==========================================================

    category_scores = (
        assessment_recent_scores
        .groupby(player_cols + ["aps_category"], as_index=False)
        .agg(
            category_aps_score=("weighted_assessment_aps", "mean"),
            n_assessments_used=("category_assessment", "nunique"),
            assessments_used=(
                "category_assessment",
                lambda x: ", ".join(sorted(x.dropna().unique()))
            )
        )
    )

    category_scores["category_aps_score"] = category_scores["category_aps_score"].round(1)
    category_scores["category_aps_band"] = (
        category_scores["category_aps_score"].apply(assign_aps_band)
    )


    # ==========================================================
    # 11. CALCULATE FINAL OVERALL APS SCORE
    # ==========================================================

    final_aps = (
        category_scores
        .groupby(player_cols, as_index=False)
        .agg(
            final_aps_score=("category_aps_score", "mean"),
            n_categories_used=("aps_category", "nunique")
        )
    )

    final_aps["final_aps_score"] = final_aps["final_aps_score"].round(1)
    final_aps["final_aps_band"] = final_aps["final_aps_score"].apply(assign_aps_band)


    # ==========================================================
    # 12. DASHBOARD-FRIENDLY FINAL TABLE
    # ==========================================================

    category_scores_wide = category_scores.pivot_table(
        index=player_cols,
        columns="aps_category",
        values="category_aps_score",
        aggfunc="mean"
    ).reset_index()

    final_dashboard_scores = final_aps.merge(
        category_scores_wide,
        on=player_cols,
        how="left"
    )


    # ==========================================================
    # 13. AUDIT TABLES
    # ==========================================================

    unmapped_assessments_audit = (
        df[df["aps_category"] == "Unmapped"]
        ["category_assessment"]
        .value_counts()
        .reset_index()
    )

    unmapped_assessments_audit.columns = [
        "unmapped_category_assessment",
        "count"
    ]

    small_sample_audit = (
        benchmark_base[
            [
                "category_assessment",
                "cohort",
                "age_group",
                "n_original_cohort",
                "combined_for_small_sample",
                "benchmark_type",
                "benchmark_cohort",
                "aps_distribution_used"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            ["category_assessment", "cohort"]
        )
    )


    # ==========================================================

    return {"attempts": scoring_df, "benchmarks": benchmark_table,
            "assessments": assessment_recent_scores, "categories": category_scores,
            "players": final_dashboard_scores, "unmapped": unmapped_assessments_audit,
            "small_samples": small_sample_audit}
