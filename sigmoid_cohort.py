"""RPS reference extracted from the supplied notebook export; see VALIDATION.md for documented edge-case safeguards."""
import numpy as np
import pandas as pd

def compare_global_vs_cohort_sigmoid_rps(
    df,
    skill_col="category_assessment",
    score_col="score",
    cohort_col="cohort",
    age_group_col="age_group",
    gender_col="gender_clean",
    timestamp_col="timestamp",
    player_id_col=None,
    span_percentile=0.75,
    min_cohort_n=30,
    skill_type_map=None,
    good_direction_map=None,
    clip_scores=True
):
    """
    RPS scoring system.

    Final RPS range:
        RPS = 30 + 69 * S(x)

    Small sample handling:
        1. Use age + gender cohort if cohort_n >= min_cohort_n
        2. If cohort_n < min_cohort_n, drop gender and use same age group
        3. If age group parameters are unavailable, use global as safety fallback

    Overall RPS weighting:
        1 attempt  = 100%
        2 attempts = 2/3 most recent, 1/3 older
        3+ attempts = 0.50, 0.30, 0.20
    """

    data = df.copy()

    # ========================================================
    # 1. Validate required columns
    # ========================================================

    required_cols = [
        skill_col,
        score_col,
        cohort_col,
        age_group_col,
        gender_col,
        timestamp_col
    ]

    if player_id_col is not None:
        required_cols.append(player_id_col)
    else:
        required_cols.extend(["first_name", "last_name"])

    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # ========================================================
    # 2. Basic cleaning
    # ========================================================

    data[score_col] = pd.to_numeric(data[score_col], errors="coerce")
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")

    data = data.dropna(
        subset=[
            skill_col,
            score_col,
            cohort_col,
            age_group_col,
            gender_col,
            timestamp_col
        ]
    ).copy()

    # ========================================================
    # 3. Define player identity columns
    # ========================================================

    if player_id_col is not None:
        player_cols = [player_id_col]
    else:
        player_cols = ["first_name", "last_name", gender_col, age_group_col]

    # ========================================================
    # 4. Maps
    # ========================================================

    if skill_type_map is None:
        skill_type_map = {}

    if good_direction_map is None:
        good_direction_map = {}

    data["skill_type"] = data[skill_col].map(skill_type_map).fillna("Normal")
    data["good_direction"] = data[skill_col].map(good_direction_map).fillna("Hi")

    # ========================================================
    # 5. Log transformation
    # Do not log-transform before calling this function.
    # ========================================================

    def safe_log(x):
        if pd.isna(x) or x <= 0:
            return np.nan
        return np.log(x)

    data["score_model"] = np.where(
        data["skill_type"].eq("Log"),
        data[score_col].apply(safe_log),
        data[score_col]
    )

    data = data.dropna(subset=["score_model"]).copy()

    # ========================================================
    # 6. Orient score direction
    # Higher score_oriented always means better.
    # ========================================================

    data["score_oriented"] = np.where(
        data["good_direction"].eq("Lo"),
        -data["score_model"],
        data["score_model"]
    )

    # ========================================================
    # 7. Sigmoid helper
    # ========================================================

    S_SPAN = 1.0986
    RPS_MIN = 30
    RPS_MAX = 99
    RPS_RANGE = RPS_MAX - RPS_MIN

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -700, 700)))

    def calculate_rps(x, mid, span):
        denom = abs(span - mid)

        if pd.isna(x) or pd.isna(mid) or pd.isna(span) or denom == 0:
            return np.nan

        z = S_SPAN * ((x - mid) / denom)
        s_x = sigmoid(z)

        score = RPS_MIN + RPS_RANGE * s_x

        if clip_scores:
            score = np.clip(score, RPS_MIN, RPS_MAX)

        return score

    # ========================================================
    # 8. Global parameters by assessment
    # ========================================================

    global_params = (
        data
        .groupby(skill_col)
        .agg(
            global_n=("score_oriented", "size"),
            global_mid=("score_oriented", "median"),
            global_span=("score_oriented", lambda x: x.quantile(span_percentile)),
            global_p25=("score_oriented", lambda x: x.quantile(0.25)),
            global_p75=("score_oriented", lambda x: x.quantile(0.75)),
            skill_type=("skill_type", "first"),
            good_direction=("good_direction", "first")
        )
        .reset_index()
    )

    # ========================================================
    # 9. Age + gender cohort parameters
    # ========================================================

    cohort_params = (
        data
        .groupby([skill_col, cohort_col])
        .agg(
            cohort_n=("score_oriented", "size"),
            cohort_mid=("score_oriented", "median"),
            cohort_span=("score_oriented", lambda x: x.quantile(span_percentile)),
            cohort_p25=("score_oriented", lambda x: x.quantile(0.25)),
            cohort_p75=("score_oriented", lambda x: x.quantile(0.75))
        )
        .reset_index()
    )

    # ========================================================
    # 10. Age-only fallback parameters
    # This combines gender within the same age group.
    # Example: 3_M + 3_F becomes age group 3.
    # ========================================================

    age_group_params = (
        data
        .groupby([skill_col, age_group_col])
        .agg(
            age_group_n=("score_oriented", "size"),
            age_group_mid=("score_oriented", "median"),
            age_group_span=("score_oriented", lambda x: x.quantile(span_percentile)),
            age_group_p25=("score_oriented", lambda x: x.quantile(0.25)),
            age_group_p75=("score_oriented", lambda x: x.quantile(0.75))
        )
        .reset_index()
    )

    # ========================================================
    # 11. Merge parameters back
    # ========================================================

    result_df = data.merge(
        global_params[
            [
                skill_col,
                "global_n",
                "global_mid",
                "global_span"
            ]
        ],
        on=skill_col,
        how="left"
    )

    result_df = result_df.merge(
        cohort_params[
            [
                skill_col,
                cohort_col,
                "cohort_n",
                "cohort_mid",
                "cohort_span",
                "cohort_p25",
                "cohort_p75"
            ]
        ],
        on=[skill_col, cohort_col],
        how="left"
    )

    result_df = result_df.merge(
        age_group_params[
            [
                skill_col,
                age_group_col,
                "age_group_n",
                "age_group_mid",
                "age_group_span",
                "age_group_p25",
                "age_group_p75"
            ]
        ],
        on=[skill_col, age_group_col],
        how="left"
    )

    # ========================================================
    # 12. Select final RPS benchmark source
    # ========================================================

    result_df["cohort_is_small"] = result_df["cohort_n"] < min_cohort_n

    result_df["rps_benchmark_source"] = np.select(
        [
            result_df["cohort_n"] >= min_cohort_n,
            result_df["cohort_n"] < min_cohort_n
        ],
        [
            "age_gender_cohort",
            "same_age_group_gender_combined"
        ],
        default="global_fallback"
    )

    result_df["rps_mid"] = np.where(
        result_df["cohort_n"] >= min_cohort_n,
        result_df["cohort_mid"],
        result_df["age_group_mid"]
    )

    result_df["rps_span"] = np.where(
        result_df["cohort_n"] >= min_cohort_n,
        result_df["cohort_span"],
        result_df["age_group_span"]
    )

    # Safety fallback only if age-group parameters are missing or invalid
    result_df["rps_benchmark_source"] = np.where(
        result_df["rps_mid"].isna() | result_df["rps_span"].isna() | (result_df["rps_mid"] == result_df["rps_span"]),
        "global_fallback",
        result_df["rps_benchmark_source"]
    )

    result_df["rps_mid"] = np.where(
        result_df["rps_benchmark_source"].eq("global_fallback"),
        result_df["global_mid"],
        result_df["rps_mid"]
    )

    result_df["rps_span"] = np.where(
        result_df["rps_benchmark_source"].eq("global_fallback"),
        result_df["global_span"],
        result_df["rps_span"]
    )

    # ========================================================
    # 13. Calculate attempt-level RPS
    # ========================================================

    result_df["rps_global_sigmoid"] = result_df.apply(
        lambda row: calculate_rps(
            row["score_oriented"],
            row["global_mid"],
            row["global_span"]
        ),
        axis=1
    )

    result_df["rps_cohort_sigmoid"] = result_df.apply(
        lambda row: calculate_rps(
            row["score_oriented"],
            row["rps_mid"],
            row["rps_span"]
        ),
        axis=1
    )

    result_df["rps_difference"] = (
        result_df["rps_cohort_sigmoid"] - result_df["rps_global_sigmoid"]
    )

    result_df["abs_rps_difference"] = result_df["rps_difference"].abs()

    # ========================================================
    # 14. Recency-weighted assessment-level RPS
    # ========================================================

    sort_cols = player_cols + [skill_col, timestamp_col]

    result_df = result_df.sort_values(
        sort_cols,
        ascending=[True] * (len(player_cols) + 1) + [False]
    ).copy()

    result_df["attempt_count"] = (
        result_df
        .groupby(player_cols + [skill_col])[skill_col]
        .transform("size")
    )

    result_df["attempt_recency_rank"] = (
        result_df
        .groupby(player_cols + [skill_col])
        .cumcount() + 1
    )

    result_df["use_for_overall_rps"] = result_df["attempt_recency_rank"] <= 3

    def assign_attempt_weight(row):
        attempt_count = row["attempt_count"]
        rank = row["attempt_recency_rank"]

        if rank > 3:
            return 0.0

        if attempt_count == 1:
            return 1.0

        if attempt_count == 2:
            return 2 / 3 if rank == 1 else 1 / 3

        if rank == 1:
            return 0.50
        elif rank == 2:
            return 0.30
        elif rank == 3:
            return 0.20

        return 0.0

    result_df["attempt_weight"] = result_df.apply(assign_attempt_weight, axis=1)

    result_df["weighted_global_rps"] = (
        result_df["rps_global_sigmoid"] * result_df["attempt_weight"]
    )

    result_df["weighted_cohort_rps"] = (
        result_df["rps_cohort_sigmoid"] * result_df["attempt_weight"]
    )

    player_assessment_rps = (
        result_df[result_df["use_for_overall_rps"]]
        .groupby(player_cols + [skill_col], as_index=False)
        .agg(
            assessment_global_rps=("weighted_global_rps", lambda s: s.sum(min_count=1)),
            assessment_cohort_rps=("weighted_cohort_rps", lambda s: s.sum(min_count=1)),
            attempts_used_for_rps=("attempt_recency_rank", "max"),
            total_attempts=("attempt_count", "max"),
            most_recent_timestamp=(timestamp_col, "max"),
            benchmark_source=("rps_benchmark_source", "first"),
            cohort_n=("cohort_n", "first"),
            age_group_n=("age_group_n", "first")
        )
    )

    # ========================================================
    # 15. Overall player RPS
    # ========================================================

    overall_player_rps = (
        player_assessment_rps
        .groupby(player_cols, as_index=False)
        .agg(
            overall_global_rps=("assessment_global_rps", "mean"),
            overall_cohort_rps=("assessment_cohort_rps", "mean"),
            number_of_assessments=(skill_col, "nunique")
        )
    )

    overall_player_rps["overall_rps_difference"] = (
        overall_player_rps["overall_cohort_rps"]
        - overall_player_rps["overall_global_rps"]
    )

    # ========================================================
    # 16. Diagnostics
    # ========================================================

    overall_comparison = pd.DataFrame({
        "metric": [
            "number_of_records",
            "number_of_skills",
            "number_of_cohorts",
            "avg_global_rps",
            "avg_cohort_rps",
            "avg_difference",
            "median_difference",
            "avg_absolute_difference",
            "correlation_global_vs_cohort",
            "pct_records_diff_gt_5",
            "pct_records_diff_gt_10",
            "pct_small_cohort_records",
            "pct_age_gender_source",
            "pct_age_group_fallback_source",
            "pct_global_fallback_source",
            "avg_overall_global_rps",
            "avg_overall_cohort_rps"
        ],
        "value": [
            len(result_df),
            result_df[skill_col].nunique(),
            result_df[cohort_col].nunique(),
            result_df["rps_global_sigmoid"].mean(),
            result_df["rps_cohort_sigmoid"].mean(),
            result_df["rps_difference"].mean(),
            result_df["rps_difference"].median(),
            result_df["abs_rps_difference"].mean(),
            result_df[["rps_global_sigmoid", "rps_cohort_sigmoid"]].corr().iloc[0, 1],
            (result_df["abs_rps_difference"] > 5).mean(),
            (result_df["abs_rps_difference"] > 10).mean(),
            result_df["cohort_is_small"].mean(),
            result_df["rps_benchmark_source"].eq("age_gender_cohort").mean(),
            result_df["rps_benchmark_source"].eq("same_age_group_gender_combined").mean(),
            result_df["rps_benchmark_source"].eq("global_fallback").mean(),
            overall_player_rps["overall_global_rps"].mean(),
            overall_player_rps["overall_cohort_rps"].mean()
        ]
    })

    skill_comparison = (
        result_df
        .groupby(skill_col)
        .agg(
            n=(score_col, "size"),
            global_rps_mean=("rps_global_sigmoid", "mean"),
            cohort_rps_mean=("rps_cohort_sigmoid", "mean"),
            avg_difference=("rps_difference", "mean"),
            median_difference=("rps_difference", "median"),
            avg_abs_difference=("abs_rps_difference", "mean"),
            pct_diff_gt_5=("abs_rps_difference", lambda x: (x > 5).mean()),
            pct_diff_gt_10=("abs_rps_difference", lambda x: (x > 10).mean()),
            min_cohort_n=("cohort_n", "min"),
            max_cohort_n=("cohort_n", "max"),
            min_age_group_n=("age_group_n", "min"),
            max_age_group_n=("age_group_n", "max")
        )
        .reset_index()
        .sort_values("avg_abs_difference", ascending=False)
    )

    cohort_comparison = (
        result_df
        .groupby(cohort_col)
        .agg(
            n=(score_col, "size"),
            global_rps_mean=("rps_global_sigmoid", "mean"),
            cohort_rps_mean=("rps_cohort_sigmoid", "mean"),
            avg_difference=("rps_difference", "mean"),
            median_difference=("rps_difference", "median"),
            avg_abs_difference=("abs_rps_difference", "mean"),
            pct_diff_gt_5=("abs_rps_difference", lambda x: (x > 5).mean()),
            pct_diff_gt_10=("abs_rps_difference", lambda x: (x > 10).mean()),
            pct_age_group_fallback=("rps_benchmark_source", lambda x: x.eq("same_age_group_gender_combined").mean()),
            pct_global_fallback=("rps_benchmark_source", lambda x: x.eq("global_fallback").mean())
        )
        .reset_index()
        .sort_values("avg_abs_difference", ascending=False)
    )

    stability_table = (
        result_df
        .groupby([skill_col, cohort_col, age_group_col], as_index=False)
        .agg(
            cohort_n=("cohort_n", "first"),
            age_group_n=("age_group_n", "first"),
            benchmark_source=("rps_benchmark_source", "first"),
            cohort_mid=("cohort_mid", "first"),
            cohort_span=("cohort_span", "first"),
            age_group_mid=("age_group_mid", "first"),
            age_group_span=("age_group_span", "first"),
            final_mid=("rps_mid", "first"),
            final_span=("rps_span", "first")
        )
        .sort_values(["benchmark_source", "cohort_n"], ascending=[True, True])
    )

    largest_changes = (
        result_df
        .sort_values("abs_rps_difference", ascending=False)
        [
            player_cols
            + [
                skill_col,
                cohort_col,
                age_group_col,
                gender_col,
                score_col,
                timestamp_col,
                "skill_type",
                "good_direction",
                "cohort_n",
                "age_group_n",
                "rps_benchmark_source",
                "rps_global_sigmoid",
                "rps_cohort_sigmoid",
                "rps_difference",
                "abs_rps_difference"
            ]
        ]
        .head(50)
    )

    monotonicity_issues = []

    for skill, temp in result_df.groupby(skill_col):
        temp = temp.sort_values("score_oriented")

        global_diff = temp["rps_global_sigmoid"].diff().dropna()

        if (global_diff < -1e-8).any():
            monotonicity_issues.append({
                "skill": skill,
                "system": "global",
                "issue": "RPS decreases when oriented score increases"
            })

        for source_group, ctemp in temp.groupby([cohort_col, "rps_benchmark_source"]):
            cohort_diff = ctemp["rps_cohort_sigmoid"].diff().dropna()

            if (cohort_diff < -1e-8).any():
                monotonicity_issues.append({
                    "skill": skill,
                    "group": source_group,
                    "system": "cohort",
                    "issue": "RPS decreases within benchmark group when oriented score increases"
                })

    monotonicity_check = pd.DataFrame(monotonicity_issues)

    diagnostics = {
        "overall_comparison": overall_comparison,
        "skill_comparison": skill_comparison,
        "cohort_comparison": cohort_comparison,
        "stability_table": stability_table,
        "largest_changes": largest_changes,
        "monotonicity_check": monotonicity_check,
        "player_assessment_rps": player_assessment_rps,
        "overall_player_rps": overall_player_rps
    }

    return result_df, global_params, cohort_params, age_group_params, diagnostics
