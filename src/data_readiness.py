import pandas as pd


MINIMUM_REQUIRED_COLUMNS = [
    "supplier_name",
    "annual_spend",
]

CLASSIFICATION_COLUMNS = [
    "category",
    "description",
]

FULL_SCORING_COLUMNS = [
    "prior_year_spend",
    "on_time_delivery_pct",
    "prior_year_otd_pct",
    "defect_rate_pct",
    "prior_year_defect_rate_pct",
    "supplier_criticality",
]

CONTEXT_COLUMNS = [
    "supplier_id",
    "invoice_date",
    "po_number",
    "business_unit",
    "cost_center",
    "buyer",
    "region",
    "country",
    "contract_status",
    "payment_terms",
    "currency",
    "gl_account",
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


def has_classification_input(
    data: pd.DataFrame,
) -> bool:
    """
    Check whether the dataset has category or description.
    """
    return any(
        column in data.columns
        for column in CLASSIFICATION_COLUMNS
    )


def evaluate_minimum_required_columns(
    data: pd.DataFrame,
) -> tuple[bool, list[str]]:
    """
    Evaluate whether the dataset has minimum fields for analysis.

    Minimum required:
    - supplier_name
    - annual_spend / spend amount
    - category or description
    """
    missing_columns = get_missing_columns(
        data,
        MINIMUM_REQUIRED_COLUMNS,
    )

    if not has_classification_input(data):
        missing_columns.append(
            "category or description"
        )

    is_ready = len(missing_columns) == 0

    return is_ready, missing_columns


def detect_input_file_type(
    data: pd.DataFrame,
) -> str:
    """
    Infer whether the file looks supplier-level or transaction-level.

    This is a heuristic. It is meant to guide readiness messaging,
    not permanently classify the source.
    """
    transaction_signals = [
        "invoice_date",
        "po_number",
        "business_unit",
        "cost_center",
        "buyer",
        "gl_account",
    ]

    transaction_signal_count = len(
        get_present_columns(
            data,
            transaction_signals,
        )
    )

    supplier_duplicate_count = 0

    if "supplier_name" in data.columns:
        supplier_duplicate_count = int(
            data["supplier_name"]
            .duplicated(keep=False)
            .sum()
        )

    if transaction_signal_count >= 2:
        return "Transaction-level spend file"

    if supplier_duplicate_count > 0:
        return "Likely multi-row supplier spend file"

    return "Supplier-level performance file"


def evaluate_analysis_capabilities(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate which analyses can run from the available columns.
    """
    minimum_ready, missing_required = (
        evaluate_minimum_required_columns(data)
    )

    has_category = "category" in data.columns
    has_description = "description" in data.columns

    has_prior_spend = "prior_year_spend" in data.columns

    has_delivery = all(
        column in data.columns
        for column in [
            "on_time_delivery_pct",
            "prior_year_otd_pct",
        ]
    )

    has_quality = all(
        column in data.columns
        for column in [
            "defect_rate_pct",
            "prior_year_defect_rate_pct",
        ]
    )

    has_criticality = (
        "supplier_criticality" in data.columns
    )

    rows = [
        {
            "capability": "Basic spend analysis",
            "status": (
                "Available"
                if minimum_ready
                else "Unavailable"
            ),
            "reason": (
                "Supplier, spend amount, and classification input found."
                if minimum_ready
                else "Missing: " + ", ".join(missing_required)
            ),
        },
        {
            "capability": "Spend classification",
            "status": (
                "Available"
                if has_category or has_description
                else "Unavailable"
            ),
            "reason": (
                "Category or description is available."
                if has_category or has_description
                else "Missing both category and description."
            ),
        },
        {
            "capability": "Supplier aggregation",
            "status": (
                "Available"
                if "supplier_name" in data.columns
                and "annual_spend" in data.columns
                else "Unavailable"
            ),
            "reason": (
                "Supplier and spend amount fields are available."
                if "supplier_name" in data.columns
                and "annual_spend" in data.columns
                else "Supplier or spend amount is missing."
            ),
        },
        {
            "capability": "Category concentration",
            "status": (
                "Available"
                if minimum_ready
                else "Unavailable"
            ),
            "reason": (
                "Spend can be grouped by uploaded or classified category."
                if minimum_ready
                else "Minimum required fields are missing."
            ),
        },
        {
            "capability": "Spend movement analysis",
            "status": (
                "Available"
                if has_prior_spend
                else "Limited"
            ),
            "reason": (
                "Prior-year spend is available."
                if has_prior_spend
                else "Prior-year spend is missing."
            ),
        },
        {
            "capability": "Delivery deterioration scoring",
            "status": (
                "Available"
                if has_delivery
                else "Limited"
            ),
            "reason": (
                "Current and prior-year OTD fields are available."
                if has_delivery
                else "Current or prior-year OTD field is missing."
            ),
        },
        {
            "capability": "Quality deterioration scoring",
            "status": (
                "Available"
                if has_quality
                else "Limited"
            ),
            "reason": (
                "Current and prior-year defect-rate fields are available."
                if has_quality
                else "Current or prior-year defect-rate field is missing."
            ),
        },
        {
            "capability": "Full supplier attention scoring",
            "status": (
                "Available"
                if (
                    minimum_ready
                    and has_prior_spend
                    and has_delivery
                    and has_quality
                    and has_criticality
                )
                else "Limited"
            ),
            "reason": (
                "Spend, performance, and criticality fields are available."
                if (
                    minimum_ready
                    and has_prior_spend
                    and has_delivery
                    and has_quality
                    and has_criticality
                )
                else "Some performance, prior-year, or criticality fields are missing."
            ),
        },
    ]

    return pd.DataFrame(rows)


def determine_analysis_status(
    minimum_ready: bool,
    capabilities: pd.DataFrame,
) -> str:
    """
    Determine overall readiness status.
    """
    if not minimum_ready:
        return "Not Ready"

    limited_count = int(
        (
            capabilities["status"]
            == "Limited"
        ).sum()
    )

    unavailable_count = int(
        (
            capabilities["status"]
            == "Unavailable"
        ).sum()
    )

    if limited_count > 0 or unavailable_count > 0:
        return "Ready with Limitations"

    return "Ready"


def create_analysis_limitations(
    analysis_status: str,
    capabilities: pd.DataFrame,
) -> list[str]:
    """
    Create human-readable limitations from capability table.
    """
    if analysis_status == "Ready":
        return [
            "No major readiness limitations detected."
        ]

    limitations = []

    limited_or_unavailable = capabilities[
        capabilities["status"].isin(
            [
                "Limited",
                "Unavailable",
            ]
        )
    ]

    for _, row in limited_or_unavailable.iterrows():
        limitations.append(
            f"{row['capability']}: {row['reason']}"
        )

    return limitations


def summarize_mapping_report(
    mapping_report: pd.DataFrame | None,
) -> dict[str, int]:
    """
    Summarize mapped, context, and unmapped columns.
    """
    if mapping_report is None or mapping_report.empty:
        return {
            "mapped_column_count": 0,
            "context_column_count": 0,
            "unmapped_column_count": 0,
            "used_in_analysis_count": 0,
        }

    mapped_column_count = int(
        (
            mapping_report["mapping_status"]
            == "Mapped"
        ).sum()
    )

    context_column_count = int(
        (
            mapping_report["column_role"]
            == "Context"
        ).sum()
    )

    unmapped_column_count = int(
        (
            mapping_report["mapping_status"]
            == "Unmapped"
        ).sum()
    )

    used_in_analysis_count = int(
        mapping_report["used_in_analysis"].sum()
    )

    return {
        "mapped_column_count": mapped_column_count,
        "context_column_count": context_column_count,
        "unmapped_column_count": unmapped_column_count,
        "used_in_analysis_count": used_in_analysis_count,
    }


def create_column_readiness_table(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a table showing required, classification, scoring,
    and context columns.
    """
    rows = []

    for column in MINIMUM_REQUIRED_COLUMNS:
        rows.append(
            {
                "column": column,
                "column_group": "Minimum required",
                "status": (
                    "Present"
                    if column in data.columns
                    else "Missing"
                ),
                "why_it_matters": (
                    "Required for basic supplier spend analysis."
                ),
            }
        )

    rows.append(
        {
            "column": "category or description",
            "column_group": "Minimum required",
            "status": (
                "Present"
                if has_classification_input(data)
                else "Missing"
            ),
            "why_it_matters": (
                "Required to classify or group spend into categories."
            ),
        }
    )

    for column in FULL_SCORING_COLUMNS:
        rows.append(
            {
                "column": column,
                "column_group": "Full scoring",
                "status": (
                    "Present"
                    if column in data.columns
                    else "Missing"
                ),
                "why_it_matters": (
                    "Improves supplier attention scoring and findings."
                ),
            }
        )

    for column in CONTEXT_COLUMNS:
        rows.append(
            {
                "column": column,
                "column_group": "Context",
                "status": (
                    "Present"
                    if column in data.columns
                    else "Missing"
                ),
                "why_it_matters": (
                    "Helpful for filtering, traceability, or future enhancements."
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
    minimum_ready, missing_required = (
        evaluate_minimum_required_columns(data)
    )

    capabilities = evaluate_analysis_capabilities(data)

    analysis_status = determine_analysis_status(
        minimum_ready,
        capabilities,
    )

    limitations = create_analysis_limitations(
        analysis_status,
        capabilities,
    )

    mapping_summary = summarize_mapping_report(
        mapping_report
    )

    input_file_type = detect_input_file_type(data)

    return {
        "row_count": len(data),
        "column_count": len(data.columns),
        "input_file_type": input_file_type,
        "analysis_status": analysis_status,
        "minimum_required_ready": minimum_ready,
        "missing_required_columns": missing_required,
        "analysis_capabilities": capabilities,
        "analysis_limitations": limitations,
        "column_readiness_table": create_column_readiness_table(data),
        **mapping_summary,
    }