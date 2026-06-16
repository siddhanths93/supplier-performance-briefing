import re

import pandas as pd


COLUMN_ALIASES = {
    "supplier_id": [
        "supplier id",
        "supplier_id",
        "vendor id",
        "vendor_id",
        "vendor number",
        "vendor no",
        "supplier number",
        "supplier no",
    ],
    "supplier_name": [
        "supplier",
        "supplier name",
        "supplier_name",
        "vendor",
        "vendor name",
        "vendor_name",
        "payee",
        "payee name",
        "merchant",
        "supplier legal name",
    ],
    "category": [
        "category",
        "spend category",
        "spend_category",
        "commodity",
        "commodity group",
        "commodity_group",
        "purchasing category",
        "gl category",
        "expense category",
    ],
    "description": [
        "description",
        "item description",
        "invoice description",
        "po description",
        "purchase description",
        "transaction description",
        "line description",
    ],
    "annual_spend": [
        "annual spend",
        "annual_spend",
        "spend",
        "amount",
        "invoice amount",
        "invoice_amount",
        "po amount",
        "po_amount",
        "total spend",
        "total_spend",
        "extended amount",
        "net amount",
    ],
    "prior_year_spend": [
        "prior year spend",
        "prior_year_spend",
        "previous year spend",
        "last year spend",
        "py spend",
        "prior spend",
    ],
    "on_time_delivery_pct": [
        "on time delivery",
        "on time delivery %",
        "on_time_delivery_pct",
        "otd",
        "otd %",
        "otd pct",
        "current otd",
    ],
    "prior_year_otd_pct": [
        "prior year otd",
        "prior year otd %",
        "prior_year_otd_pct",
        "previous year otd",
        "last year otd",
        "py otd",
    ],
    "defect_rate_pct": [
        "defect rate",
        "defect rate %",
        "defect_rate_pct",
        "quality defect rate",
        "current defect rate",
    ],
    "prior_year_defect_rate_pct": [
        "prior year defect rate",
        "prior year defect rate %",
        "prior_year_defect_rate_pct",
        "previous year defect rate",
        "last year defect rate",
        "py defect rate",
    ],
    "lead_time_days": [
        "lead time",
        "lead time days",
        "lead_time_days",
        "avg lead time",
        "average lead time",
    ],
    "region": [
        "region",
        "supplier region",
        "geo",
        "geography",
        "country region",
    ],
    "contract_status": [
        "contract status",
        "contract_status",
        "contract",
        "agreement status",
    ],
    "supplier_criticality": [
        "supplier criticality",
        "supplier_criticality",
        "criticality",
        "business criticality",
        "critical supplier",
    ],
    "invoice_date": [
        "invoice date",
        "invoice_date",
        "transaction date",
        "posting date",
        "date",
        "po date",
    ],
}


def normalize_column_name(column_name: str) -> str:
    """
    Normalize a column name for matching.

    Examples:
    'Vendor Name' -> 'vendor name'
    'vendor_name' -> 'vendor name'
    'Vendor-Name' -> 'vendor name'
    """
    normalized_name = str(column_name).strip().lower()

    normalized_name = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized_name,
    )

    normalized_name = re.sub(
        r"\s+",
        " ",
        normalized_name,
    ).strip()

    return normalized_name


def build_alias_lookup() -> dict[str, str]:
    """
    Build a lookup from normalized alias to canonical column name.
    """
    alias_lookup = {}

    for canonical_column, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            alias_lookup[normalized_alias] = canonical_column

    return alias_lookup


def map_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rename recognized uploaded columns to canonical names.

    Returns:
    - mapped DataFrame
    - mapping report DataFrame
    """
    data = data.copy()

    alias_lookup = build_alias_lookup()

    rename_mapping = {}
    report_rows = []

    used_canonical_columns = set()

    for original_column in data.columns:
        normalized_column = normalize_column_name(
            original_column
        )

        canonical_column = alias_lookup.get(
            normalized_column
        )

        if (
            canonical_column is not None
            and canonical_column not in used_canonical_columns
        ):
            rename_mapping[original_column] = canonical_column
            used_canonical_columns.add(canonical_column)

            report_rows.append(
                {
                    "original_column": original_column,
                    "mapped_column": canonical_column,
                    "mapping_status": "Mapped",
                }
            )

        elif canonical_column is not None:
            report_rows.append(
                {
                    "original_column": original_column,
                    "mapped_column": canonical_column,
                    "mapping_status": (
                        "Duplicate ignored"
                    ),
                }
            )

        else:
            report_rows.append(
                {
                    "original_column": original_column,
                    "mapped_column": "",
                    "mapping_status": "Unmapped",
                }
            )

    mapped_data = data.rename(
        columns=rename_mapping
    )

    mapping_report = pd.DataFrame(report_rows)

    return mapped_data, mapping_report