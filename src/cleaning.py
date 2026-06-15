import re

import pandas as pd


TEXT_COLUMNS = [
    "supplier_name",
    "category",
    "region",
    "contract_status",
    "supplier_criticality",
]

NUMERIC_COLUMNS = [
    "annual_spend",
    "prior_year_spend",
    "on_time_delivery_pct",
    "prior_year_otd_pct",
    "defect_rate_pct",
    "prior_year_defect_rate_pct",
    "lead_time_days",
]


def standardize_column_names(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convert column names into consistent snake_case format.

    Example:
    'Annual Spend' becomes 'annual_spend'.
    """
    data = data.copy()

    cleaned_columns = []

    for column in data.columns:
        cleaned_column = str(column).strip().lower()
        cleaned_column = re.sub(
            r"[^a-z0-9]+",
            "_",
            cleaned_column,
        )
        cleaned_column = cleaned_column.strip("_")

        cleaned_columns.append(cleaned_column)

    data.columns = cleaned_columns

    return data


def clean_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Trim spaces and standardize text formatting.
    """
    data = data.copy()

    for column in TEXT_COLUMNS:
        if column not in data.columns:
            continue

        data[column] = (
            data[column]
            .astype("string")
            .str.strip()
            .replace(
                {
                    "": pd.NA,
                    "N/A": pd.NA,
                    "None": pd.NA,
                    "null": pd.NA,
                }
            )
        )

    return data


def standardize_category_values(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize category labels using title case.
    """
    data = data.copy()

    if "category" in data.columns:
        data["category"] = (
            data["category"]
            .astype("string")
            .str.strip()
            .str.title()
        )

    return data


def standardize_region_values(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize common region abbreviations and label variants.
    """
    data = data.copy()

    if "region" not in data.columns:
        return data

    region_mapping = {
        "NA": "North America",
        "NORTH AMERICA": "North America",
        "EU": "Europe",
        "EUROPE": "Europe",
        "APAC": "Asia Pacific",
        "ASIA PACIFIC": "Asia Pacific",
        "LATAM": "Latin America",
        "LATIN AMERICA": "Latin America",
    }

    data["region"] = (
        data["region"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(region_mapping)
    )

    return data


def standardize_contract_status(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize contract-status values.
    """
    data = data.copy()

    if "contract_status" not in data.columns:
        return data

    status_mapping = {
        "ACTIVE": "Active",
        "EXPIRED": "Expired",
        "NO CONTRACT": "No Contract",
        "NO-CONTRACT": "No Contract",
        "UNDER REVIEW": "Under Review",
    }

    data["contract_status"] = (
        data["contract_status"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(status_mapping)
    )

    return data


def standardize_criticality(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize supplier-criticality values.
    """
    data = data.copy()

    if "supplier_criticality" not in data.columns:
        return data

    criticality_mapping = {
        "LOW": "Low",
        "MEDIUM": "Medium",
        "HIGH": "High",
    }

    data["supplier_criticality"] = (
        data["supplier_criticality"]
        .astype("string")
        .str.strip()
        .str.upper()
        .replace(criticality_mapping)
    )

    return data


def convert_numeric_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert known numeric fields to numeric data types.

    Invalid values become missing values rather than causing the
    application to crash.
    """
    data = data.copy()

    for column in NUMERIC_COLUMNS:
        if column not in data.columns:
            continue

        data[column] = (
            data[column]
            .astype("string")
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    return data


def clean_supplier_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run the complete cleaning pipeline.
    """
    cleaned_data = standardize_column_names(data)
    cleaned_data = clean_text_columns(cleaned_data)
    cleaned_data = standardize_category_values(
        cleaned_data
    )
    cleaned_data = standardize_region_values(
        cleaned_data
    )
    cleaned_data = standardize_contract_status(
        cleaned_data
    )
    cleaned_data = standardize_criticality(
        cleaned_data
    )
    cleaned_data = convert_numeric_columns(
        cleaned_data
    )

    return cleaned_data