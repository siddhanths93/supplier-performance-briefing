import pandas as pd


NUMERIC_COLUMNS = [
    "annual_spend",
    "prior_year_spend",
    "on_time_delivery_pct",
    "prior_year_otd_pct",
    "defect_rate_pct",
    "prior_year_defect_rate_pct",
    "lead_time_days",
]


def calculate_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate missing-value counts and percentages by column.
    """
    missing_summary = pd.DataFrame(
        {
            "column": data.columns,
            "missing_count": data.isna().sum().values,
            "missing_pct": (
                data.isna().mean().values * 100
            ).round(2),
        }
    )

    return missing_summary.sort_values(
        by="missing_pct",
        ascending=False,
    ).reset_index(drop=True)


def count_exact_duplicate_rows(data: pd.DataFrame) -> int:
    """
    Count rows that are exact duplicates of earlier rows.
    """
    return int(data.duplicated().sum())


def count_duplicate_supplier_names(data: pd.DataFrame) -> int:
    """
    Count repeated supplier names.

    This checks exact text matches only.
    Fuzzy matching will be added later.
    """
    if "supplier_name" not in data.columns:
        return 0

    return int(
        data["supplier_name"]
        .duplicated(keep=False)
        .sum()
    )


def count_invalid_spend_values(data: pd.DataFrame) -> int:
    """
    Count rows where annual spend is missing, zero, or negative.
    """
    if "annual_spend" not in data.columns:
        return 0

    invalid_rows = (
        data["annual_spend"].isna()
        | (data["annual_spend"] <= 0)
    )

    return int(invalid_rows.sum())


def count_invalid_percentage_values(
    data: pd.DataFrame,
) -> int:
    """
    Count percentage values outside the valid 0 to 100 range.
    """
    percentage_columns = [
        "on_time_delivery_pct",
        "prior_year_otd_pct",
        "defect_rate_pct",
        "prior_year_defect_rate_pct",
    ]

    invalid_count = 0

    for column in percentage_columns:
        if column not in data.columns:
            continue

        invalid_rows = (
            data[column].notna()
            & (
                (data[column] < 0)
                | (data[column] > 100)
            )
        )

        invalid_count += int(invalid_rows.sum())

    return invalid_count


def create_data_quality_summary(
    data: pd.DataFrame,
) -> dict[str, int | float]:
    """
    Create a high-level data-quality summary.
    """
    total_cells = data.shape[0] * data.shape[1]
    missing_cells = int(data.isna().sum().sum())

    missing_cell_pct = (
        (missing_cells / total_cells) * 100
        if total_cells > 0
        else 0.0
    )

    return {
        "rows": len(data),
        "columns": len(data.columns),
        "missing_cells": missing_cells,
        "missing_cell_pct": round(
            missing_cell_pct,
            2,
        ),
        "exact_duplicate_rows": count_exact_duplicate_rows(
            data
        ),
        "duplicate_supplier_name_rows": (
            count_duplicate_supplier_names(data)
        ),
        "invalid_spend_rows": count_invalid_spend_values(
            data
        ),
        "invalid_percentage_values": (
            count_invalid_percentage_values(data)
        ),
    }