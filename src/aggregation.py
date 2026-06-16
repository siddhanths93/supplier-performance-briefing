import pandas as pd


BASE_GROUPING_COLUMNS = [
    "supplier_name",
    "category",
]


TIME_GRAIN_COLUMNS = {
    None: [],
    "year": ["invoice_year"],
    "quarter": ["invoice_quarter"],
    "month": ["invoice_month"],
}


SUM_COLUMNS = [
    "annual_spend",
    "prior_year_spend",
]


MEAN_COLUMNS = [
    "on_time_delivery_pct",
    "prior_year_otd_pct",
    "defect_rate_pct",
    "prior_year_defect_rate_pct",
    "lead_time_days",
    "classification_score",
]


FIRST_VALUE_COLUMNS = [
    "supplier_id",
    "taxonomy_level_1",
    "taxonomy_level_2",
    "taxonomy_code",
    "analysis_category",
    "classification_source",
    "classification_confidence",
    "classification_reason",
    "needs_classification_review",
    "supplier_criticality",
    "contract_status",
    "region",
    "country",
]


def get_existing_columns(
    data: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    """
    Return only columns that exist in the DataFrame.
    """
    return [
        column
        for column in columns
        if column in data.columns
    ]


def get_grouping_columns(
    time_grain: str | None = None,
) -> list[str]:
    """
    Return grouping columns for supplier/category aggregation.

    time_grain options:
    - None
    - year
    - quarter
    - month
    """
    if time_grain not in TIME_GRAIN_COLUMNS:
        raise ValueError(
            "Invalid time_grain. Use one of: None, year, quarter, month."
        )

    return BASE_GROUPING_COLUMNS + TIME_GRAIN_COLUMNS[time_grain]


def first_non_null(series: pd.Series):
    """
    Return first non-null value from a pandas Series.
    """
    non_null_values = series.dropna()

    if non_null_values.empty:
        return pd.NA

    return non_null_values.iloc[0]


def should_aggregate_supplier_data(
    data: pd.DataFrame,
    time_grain: str | None = None,
) -> bool:
    """
    Determine whether data has multiple rows for the same grouping level.
    """
    grouping_columns = get_grouping_columns(
        time_grain=time_grain
    )

    existing_grouping_columns = get_existing_columns(
        data,
        grouping_columns,
    )

    if existing_grouping_columns != grouping_columns:
        return False

    duplicate_count = int(
        data.duplicated(
            subset=grouping_columns,
            keep=False,
        ).sum()
    )

    return duplicate_count > 0


def aggregate_supplier_category_data(
    data: pd.DataFrame,
    time_grain: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate supplier data to supplier/category level.

    Optional time_grain allows aggregation by:
    - supplier/category
    - supplier/category/year
    - supplier/category/quarter
    - supplier/category/month
    """
    data = data.copy()

    grouping_columns = get_grouping_columns(
        time_grain=time_grain
    )

    missing_grouping_columns = [
        column
        for column in grouping_columns
        if column not in data.columns
    ]

    if missing_grouping_columns:
        raise ValueError(
            "Cannot aggregate supplier data. Missing columns: "
            + ", ".join(missing_grouping_columns)
        )

    aggregation_rules = {}

    for column in get_existing_columns(
        data,
        SUM_COLUMNS,
    ):
        aggregation_rules[column] = "sum"

    for column in get_existing_columns(
        data,
        MEAN_COLUMNS,
    ):
        aggregation_rules[column] = "mean"

    for column in get_existing_columns(
        data,
        FIRST_VALUE_COLUMNS,
    ):
        if column not in grouping_columns:
            aggregation_rules[column] = first_non_null

    aggregated_data = (
        data.groupby(
            grouping_columns,
            dropna=False,
        )
        .agg(aggregation_rules)
        .reset_index()
    )

    transaction_counts = (
        data.groupby(
            grouping_columns,
            dropna=False,
        )
        .size()
        .reset_index(name="transaction_count")
    )

    aggregated_data = aggregated_data.merge(
        transaction_counts,
        on=grouping_columns,
        how="left",
    )

    return aggregated_data


def create_aggregation_summary(
    before_data: pd.DataFrame,
    after_data: pd.DataFrame,
    was_aggregated: bool,
    time_grain: str | None = None,
) -> dict:
    """
    Create summary information about aggregation.
    """
    return {
        "was_aggregated": was_aggregated,
        "time_grain": time_grain or "none",
        "input_rows": len(before_data),
        "output_rows": len(after_data),
        "rows_reduced": len(before_data) - len(after_data),
        "input_supplier_count": (
            before_data["supplier_name"].nunique()
            if "supplier_name" in before_data.columns
            else 0
        ),
        "output_supplier_count": (
            after_data["supplier_name"].nunique()
            if "supplier_name" in after_data.columns
            else 0
        ),
    }