import numpy as np
import pandas as pd


def calculate_spend_metrics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate supplier-level spend metrics.

    Metrics added:
    - spend_change_amount
    - spend_change_pct
    - total_spend_share_pct
    - category_spend
    - category_spend_share_pct
    """
    data = data.copy()

    if "annual_spend" not in data.columns:
        raise ValueError(
            "annual_spend is required to calculate spend metrics."
        )

    total_spend = data["annual_spend"].sum()

    data["spend_change_amount"] = (
        data["annual_spend"]
        - data["prior_year_spend"]
    )

    data["spend_change_pct"] = np.where(
        data["prior_year_spend"].notna()
        & (data["prior_year_spend"] != 0),
        (
            data["spend_change_amount"]
            / data["prior_year_spend"]
        )
        * 100,
        np.nan,
    )

    data["total_spend_share_pct"] = np.where(
        total_spend != 0,
        (data["annual_spend"] / total_spend) * 100,
        np.nan,
    )

    data["category_spend"] = (
        data.groupby("category")["annual_spend"]
        .transform("sum")
    )

    data["category_spend_share_pct"] = np.where(
        data["category_spend"] != 0,
        (
            data["annual_spend"]
            / data["category_spend"]
        )
        * 100,
        np.nan,
    )

    return data


def calculate_performance_changes(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate supplier-level delivery and quality changes.

    Positive delivery change means improvement.
    Positive defect-rate change means deterioration.
    """
    data = data.copy()

    data["otd_change_pct_points"] = (
        data["on_time_delivery_pct"]
        - data["prior_year_otd_pct"]
    )

    data["defect_rate_change_pct_points"] = (
        data["defect_rate_pct"]
        - data["prior_year_defect_rate_pct"]
    )

    return data


def add_supplier_performance_flags(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add transparent rule-based supplier performance flags.
    """
    data = data.copy()

    data["delivery_deterioration_flag"] = (
        data["otd_change_pct_points"] <= -5
    )

    data["quality_deterioration_flag"] = (
        data["defect_rate_change_pct_points"] >= 2
    )

    data["high_spend_exposure_flag"] = (
        data["category_spend_share_pct"] >= 20
    )

    data["missing_performance_data_flag"] = (
        data[
            [
                "on_time_delivery_pct",
                "prior_year_otd_pct",
                "defect_rate_pct",
                "prior_year_defect_rate_pct",
            ]
        ]
        .isna()
        .any(axis=1)
    )

    return data


def calculate_supplier_metrics(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the complete supplier-level metrics pipeline.
    """
    supplier_metrics = calculate_spend_metrics(data)

    supplier_metrics = calculate_performance_changes(
        supplier_metrics
    )

    supplier_metrics = add_supplier_performance_flags(
        supplier_metrics
    )

    return supplier_metrics