import numpy as np
import pandas as pd


ATTENTION_SCORE_WEIGHTS = {
    "spend_exposure_score": 0.30,
    "delivery_risk_score": 0.25,
    "quality_risk_score": 0.25,
    "criticality_score": 0.20,
}


CRITICALITY_SCORE_MAPPING = {
    "Low": 25,
    "Medium": 60,
    "High": 100,
}


def min_max_scale(
    series: pd.Series,
) -> pd.Series:
    """
    Scale numeric values to a 0-100 range.

    If all valid values are identical, valid rows receive 50.
    Missing values remain missing.
    """
    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    minimum_value = numeric_series.min()
    maximum_value = numeric_series.max()

    if pd.isna(minimum_value) or pd.isna(maximum_value):
        return pd.Series(
            np.nan,
            index=series.index,
            dtype="float64",
        )

    if minimum_value == maximum_value:
        scaled_series = pd.Series(
            50.0,
            index=series.index,
        )

        scaled_series[
            numeric_series.isna()
        ] = np.nan

        return scaled_series

    return (
        (
            numeric_series - minimum_value
        )
        / (
            maximum_value - minimum_value
        )
        * 100
    )


def calculate_spend_exposure_score(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Score supplier spend exposure within its category.

    Higher category spend share produces a higher score.
    """
    return min_max_scale(
        data["category_spend_share_pct"]
    )


def calculate_delivery_risk_score(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Score delivery risk.

    More negative OTD change means greater deterioration and risk.
    """
    delivery_deterioration = (
        -data["otd_change_pct_points"]
    )

    delivery_deterioration = (
        delivery_deterioration.clip(lower=0)
    )

    return min_max_scale(
        delivery_deterioration
    )


def calculate_quality_risk_score(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Score quality risk.

    A larger increase in defect rate produces a higher score.
    """
    quality_deterioration = (
        data["defect_rate_change_pct_points"]
        .clip(lower=0)
    )

    return min_max_scale(
        quality_deterioration
    )


def calculate_criticality_score(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Convert supplier criticality labels to numeric scores.
    """
    return data["supplier_criticality"].map(
        CRITICALITY_SCORE_MAPPING
    )


def calculate_data_confidence(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate the percentage of scoring inputs available.

    This score is separate from the attention score.
    Missing data lowers confidence but does not automatically
    increase or decrease supplier risk.
    """
    score_input_columns = [
        "category_spend_share_pct",
        "otd_change_pct_points",
        "defect_rate_change_pct_points",
        "supplier_criticality",
    ]

    available_input_count = (
        data[score_input_columns]
        .notna()
        .sum(axis=1)
    )

    total_input_count = len(
        score_input_columns
    )

    return (
        available_input_count
        / total_input_count
        * 100
    ).round(1)


def calculate_weighted_attention_score(
    data: pd.DataFrame,
) -> pd.Series:
    """
    Calculate a weighted supplier attention score.

    Missing score components are excluded and the remaining
    weights are rebalanced for that supplier.
    """
    score_columns = list(
        ATTENTION_SCORE_WEIGHTS.keys()
    )

    weighted_score_sum = pd.Series(
        0.0,
        index=data.index,
    )

    available_weight_sum = pd.Series(
        0.0,
        index=data.index,
    )

    for column in score_columns:
        weight = ATTENTION_SCORE_WEIGHTS[
            column
        ]

        available_rows = data[
            column
        ].notna()

        weighted_score_sum.loc[
            available_rows
        ] += (
            data.loc[
                available_rows,
                column,
            ]
            * weight
        )

        available_weight_sum.loc[
            available_rows
        ] += weight

    attention_score = (
        weighted_score_sum
        / available_weight_sum.replace(
            0,
            np.nan,
        )
    )

    return attention_score.round(1)


def assign_attention_level(
    attention_score: pd.Series,
) -> pd.Series:
    """
    Convert numeric scores into management-attention levels.
    """
    conditions = [
        attention_score >= 70,
        attention_score >= 45,
    ]

    choices = [
        "High",
        "Medium",
    ]

    return pd.Series(
        np.select(
            conditions,
            choices,
            default="Low",
        ),
        index=attention_score.index,
    )


def add_supplier_attention_scores(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add score components, final score, level, and confidence.
    """
    data = data.copy()

    data["spend_exposure_score"] = (
        calculate_spend_exposure_score(data)
    )

    data["delivery_risk_score"] = (
        calculate_delivery_risk_score(data)
    )

    data["quality_risk_score"] = (
        calculate_quality_risk_score(data)
    )

    data["criticality_score"] = (
        calculate_criticality_score(data)
    )

    data["data_confidence_pct"] = (
        calculate_data_confidence(data)
    )

    data["supplier_attention_score"] = (
        calculate_weighted_attention_score(
            data
        )
    )

    data["attention_level"] = (
        assign_attention_level(
            data["supplier_attention_score"]
        )
    )

    return data