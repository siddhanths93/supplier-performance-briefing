import pandas as pd


REQUIRED_COLUMNS = [
    "supplier_name",
    "annual_spend",
]

CONDITIONAL_REQUIRED_COLUMNS = [
    "category",
    "description",
]

RECOMMENDED_COLUMNS = [
    "prior_year_spend",
    "on_time_delivery_pct",
    "prior_year_otd_pct",
    "defect_rate_pct",
    "prior_year_defect_rate_pct",
    "supplier_criticality",
    "contract_status",
    "region",
]

OPTIONAL_COLUMNS = [
    "supplier_id",
    "invoice_date",
    "lead_time_days",
]


def get_present_columns(
    data: pd.DataFrame,
    expected_columns: list[str],
) -> list[str]:
    """
    Return expected columns that exist in the dataset.
    """
    return [
        column
        for column in expected_columns
        if column in data.columns
    ]


def get_missing_columns(
    data: pd.DataFrame,
    expected_columns: list[str],
) -> list[str]:
    """
    Return expected columns that are missing from the dataset.
    """
    return [
        column
        for column in expected_columns
        if column not in data.columns
    ]


def evaluate_required_columns(
    data: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """
    Evaluate whether the dataset has minimum required fields.

    Required:
    - supplier_name
    - annual_spend
    - at least one of category or description
    """
    missing_columns = get_missing_columns(
        data,
        REQUIRED_COLUMNS,
    )

    has_category_or_description = any(
        column in data.columns
        for column in CONDITIONAL_REQUIRED_COLUMNS
    )

    if not has_category_or_description:
        missing_columns.append(
            "category or description"
        )

    is_ready = len(missing_columns) == 0

    return is_ready, missing_columns


def determine_analysis_status(
    required_columns_ready: bool,
    recommended_missing: list[str],
) -> str:
    """
    Determine overall readiness status.
    """
    if not required_columns_ready:
        return "Not Ready"

    if recommended_missing:
        return "Ready with Limitations"

    return "Ready"


def create_analysis_limitations(
    required_columns_ready: bool,
    missing_required: list[str],
    missing_recommended: list[str],
) -> list[str]:
    """
    Create human-readable analysis limitations.
    """
    limitations = []

    if not required_columns_ready:
        limitations.append(
            "The file is missing minimum required fields, so analytics cannot run."
        )

    if "category or description" in missing_required:
        limitations.append(
            "The file needs either a category column or a description column so spend can be classified."
        )

    if "prior_year_spend" in missing_recommended:
        limitations.append(
            "Spend change analysis will be limited because prior-year spend is missing."
        )

    missing_delivery_columns = {
        "on_time_delivery_pct",
        "prior_year_otd_pct",
    }.intersection(missing_recommended)

    if missing_delivery_columns:
        limitations.append(
            "Delivery deterioration scoring will be limited because current or prior-year delivery data is missing."
        )

    missing_quality_columns = {
        "defect_rate_pct",
        "prior_year_defect_rate_pct",
    }.intersection(missing_recommended)

    if missing_quality_columns:
        limitations.append(
            "Quality deterioration scoring will be limited because current or prior-year defect data is missing."
        )

    if "supplier_criticality" in missing_recommended:
        limitations.append(
            "Supplier criticality scoring will be limited because supplier criticality is missing."
        )

    if not limitations:
        limitations.append(
            "No major readiness limitations detected."
        )

    return limitations


def create_column_readiness_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a table showing required, recommended, and optional columns.
    """
    rows = []

    for column in REQUIRED_COLUMNS:
        rows.append(
            {
                "column": column,
                "requirement_level": "Required",
                "status": (
                    "Present"
                    if column in data.columns
                    else "Missing"
                ),
            }
        )

    rows.append(
        {
            "column": "category or description",
            "requirement_level": "Required",
            "status": (
                "Present"
                if any(
                    column in data.columns
                    for column in CONDITIONAL_REQUIRED_COLUMNS
                )
                else "Missing"
            ),
        }
    )

    for column in RECOMMENDED_COLUMNS:
        rows.append(
            {
                "column": column,
                "requirement_level": "Recommended",
                "status": (
                    "Present"
                    if column in data.columns
                    else "Missing"
                ),
            }
        )

    for column in OPTIONAL_COLUMNS:
        rows.append(
            {
                "column": column,
                "requirement_level": "Optional",
                "status": (
                    "Present"
                    if column in data.columns
                    else "Missing"
                ),
            }
        )

    return pd.DataFrame(rows)


def create_data_readiness_report(
    data: pd.DataFrame,
    mapping_report: pd.DataFrame | None = None,
) -> dict:
    """
    Create a complete readiness report for uploaded supplier data.
    """
    required_ready, missing_required = (
        evaluate_required_columns(data)
    )

    missing_recommended = get_missing_columns(
        data,
        RECOMMENDED_COLUMNS,
    )

    present_recommended = get_present_columns(
        data,
        RECOMMENDED_COLUMNS,
    )

    present_optional = get_present_columns(
        data,
        OPTIONAL_COLUMNS,
    )

    analysis_status = determine_analysis_status(
        required_ready,
        missing_recommended,
    )

    limitations = create_analysis_limitations(
        required_columns_ready=required_ready,
        missing_required=missing_required,
        missing_recommended=missing_recommended,
    )

    unmapped_column_count = 0

    if mapping_report is not None and not mapping_report.empty:
        unmapped_column_count = int(
            (
                mapping_report["mapping_status"]
                == "Unmapped"
            ).sum()
        )

    return {
        "row_count": len(data),
        "column_count": len(data.columns),
        "required_columns_ready": required_ready,
        "missing_required_columns": missing_required,
        "recommended_columns_present": present_recommended,
        "recommended_columns_missing": missing_recommended,
        "optional_columns_present": present_optional,
        "analysis_status": analysis_status,
        "analysis_limitations": limitations,
        "unmapped_column_count": unmapped_column_count,
        "column_readiness_table": create_column_readiness_table(data),
    }