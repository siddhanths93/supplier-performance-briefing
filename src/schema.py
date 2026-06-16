import pandas as pd


OPTIONAL_ANALYSIS_COLUMNS = [
    "prior_year_spend",
    "on_time_delivery_pct",
    "prior_year_otd_pct",
    "defect_rate_pct",
    "prior_year_defect_rate_pct",
    "supplier_criticality",
    "contract_status",
    "region",
    "country",
    "lead_time_days",
    "invoice_date",
    "invoice_year",
    "invoice_quarter",
    "invoice_month",
    "transaction_count",
]


def ensure_optional_analysis_columns(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add optional analysis columns if they are missing.

    This prevents downstream scoring, metrics, exports, or charts from
    failing when the uploaded file does not include every recommended field.
    """
    data = data.copy()

    for column in OPTIONAL_ANALYSIS_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA

    return data