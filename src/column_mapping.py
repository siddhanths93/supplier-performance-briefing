import re

import pandas as pd


COLUMN_ROLES = {
    "supplier_id": "Context",
    "supplier_name": "Required",
    "category": "Classification",
    "description": "Classification",
    "annual_spend": "Required",
    "prior_year_spend": "Full scoring",
    "on_time_delivery_pct": "Full scoring",
    "prior_year_otd_pct": "Full scoring",
    "defect_rate_pct": "Full scoring",
    "prior_year_defect_rate_pct": "Full scoring",
    "supplier_criticality": "Full scoring",
    "contract_status": "Context",
    "region": "Context",
    "country": "Context",
    "invoice_date": "Context",
    "po_number": "Context",
    "business_unit": "Context",
    "cost_center": "Context",
    "buyer": "Context",
    "payment_terms": "Context",
    "gl_account": "Context",
    "currency": "Context",
    "lead_time_days": "Context",
}


ANALYSIS_USED_COLUMNS = {
    "supplier_id",
    "supplier_name",
    "category",
    "description",
    "annual_spend",
    "prior_year_spend",
    "on_time_delivery_pct",
    "prior_year_otd_pct",
    "defect_rate_pct",
    "prior_year_defect_rate_pct",
    "supplier_criticality",
    "contract_status",
    "region",
    "lead_time_days",
}


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
        "vendor code",
        "supplier code",
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
        "vendor legal name",
    ],
    "category": [
        "category",
        "spend category",
        "spend_category",
        "commodity",
        "commodity group",
        "commodity_group",
        "purchasing category",
        "procurement category",
        "gl category",
        "expense category",
        "category name",
        "material group",
        "item category",
        "category l1",
        "category_l1",
        "category level 1",
        "category_level_1",
        "category l2",
        "category_l2",
        "category level 2",
        "category_level_2",
    ],
    "description": [
        "description",
        "item description",
        "invoice description",
        "po description",
        "purchase description",
        "transaction description",
        "line description",
        "commodity description",
        "material description",
        "product description",
        "service description",
        "item_description",
        "item description",
        "transaction description",
        "line item description",
        "material description",
        "service description",
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
        "transaction amount",
        "line amount",
        "actual spend",
        "spend amount",
        "spend amount usd",
        "spend_amount_usd",
        "amount usd",
        "usd amount",
        "usd spend",
        "total spend usd",
        "invoice amount usd",
        "transaction amount usd",
        "original currency amount",
        "original_currency_amount",
    ],
    "prior_year_spend": [
        "prior year spend",
        "prior_year_spend",
        "previous year spend",
        "last year spend",
        "py spend",
        "prior spend",
        "previous spend",
    ],
    "on_time_delivery_pct": [
        "on time delivery",
        "on time delivery %",
        "on_time_delivery_pct",
        "otd",
        "otd %",
        "otd pct",
        "current otd",
        "current otd %",
        "on time delivery pct",
    ],
    "prior_year_otd_pct": [
        "prior year otd",
        "prior year otd %",
        "prior_year_otd_pct",
        "previous year otd",
        "last year otd",
        "py otd",
        "prior otd",
        "previous otd %",
    ],
    "defect_rate_pct": [
        "defect rate",
        "defect rate %",
        "defect_rate_pct",
        "quality defect rate",
        "current defect rate",
        "current defect rate %",
        "defect pct",
    ],
    "prior_year_defect_rate_pct": [
        "prior year defect rate",
        "prior year defect rate %",
        "prior_year_defect_rate_pct",
        "previous year defect rate",
        "last year defect rate",
        "py defect rate",
        "prior defect rate",
    ],
    "lead_time_days": [
        "lead time",
        "lead time days",
        "lead_time_days",
        "avg lead time",
        "average lead time",
        "average lead time days",
    ],
    "region": [
        "region",
        "supplier region",
        "geo",
        "geography",
        "country region",
        "market region",
    ],
    "country": [
        "country",
        "supplier country",
        "vendor country",
        "ship from country",
    ],
    "contract_status": [
        "contract status",
        "contract_status",
        "contract",
        "agreement status",
        "contract coverage",
    ],
    "supplier_criticality": [
        "supplier criticality",
        "supplier_criticality",
        "criticality",
        "business criticality",
        "critical supplier",
        "supplier risk tier",
    ],
    "invoice_date": [
        "invoice date",
        "invoice_date",
        "transaction date",
        "posting date",
        "date",
        "po date",
        "document date",
    ],
    "po_number": [
        "po number",
        "po_number",
        "purchase order",
        "purchase order number",
        "purchase_order",
        "po",
    ],
    "business_unit": [
        "business unit",
        "business_unit",
        "division",
        "department",
        "operating unit",
        "org unit",
    ],
    "cost_center": [
        "cost center",
        "cost_center",
        "cost centre",
        "department code",
        "cost center code",
    ],
    "buyer": [
        "buyer",
        "buyer name",
        "purchasing owner",
        "category manager",
        "sourcing owner",
        "procurement owner",
    ],
    "payment_terms": [
        "payment terms",
        "payment_terms",
        "terms",
        "supplier terms",
        "vendor terms",
    ],
    "currency": [
        "currency",
        "transaction currency",
        "document currency",
        "invoice currency",
        "original currency",
    ],
    "gl_account": [
        "gl account",
        "gl_account",
        "general ledger",
        "account code",
        "expense account",
        "ledger account",
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


def get_column_role(canonical_column: str | None) -> str:
    """
    Return the role of a mapped canonical column.
    """
    if canonical_column is None:
        return "Unmapped"

    return COLUMN_ROLES.get(canonical_column, "Unmapped")


def is_column_used_in_analysis(canonical_column: str | None) -> bool:
    """
    Return whether a canonical column is currently used by analytics.
    """
    if canonical_column is None:
        return False

    return canonical_column in ANALYSIS_USED_COLUMNS


def map_columns(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rename recognized uploaded columns to canonical names.

    Extra columns are preserved and reported.

    Returns:
    - mapped DataFrame
    - mapping report DataFrame
    """
    data = data.copy()

    alias_lookup = build_alias_lookup()

    rename_mapping = {}
    report_rows = []

    used_canonical_columns = set()

    preferred_original_columns = {
        "annual_spend": [
            "spend amount usd",
            "spend_amount_usd",
            "total spend usd",
            "invoice amount usd",
            "transaction amount usd",
            "spend amount",
            "invoice amount",
            "amount",
            "original currency amount",
            "original_currency_amount",
        ]
    }

    for original_column in data.columns:
        normalized_column = normalize_column_name(
            original_column
        )

        canonical_column = alias_lookup.get(
            normalized_column
        )

        if canonical_column is None:
            report_rows.append(
                {
                    "original_column": original_column,
                    "normalized_column": normalized_column,
                    "mapped_column": "",
                    "column_role": "Unmapped",
                    "used_in_analysis": False,
                    "mapping_status": "Unmapped",
                }
            )
            continue

        if canonical_column not in used_canonical_columns:
            rename_mapping[original_column] = canonical_column
            used_canonical_columns.add(canonical_column)

            report_rows.append(
                {
                    "original_column": original_column,
                    "normalized_column": normalized_column,
                    "mapped_column": canonical_column,
                    "column_role": get_column_role(canonical_column),
                    "used_in_analysis": is_column_used_in_analysis(
                        canonical_column
                    ),
                    "mapping_status": "Mapped",
                }
            )
            continue

        # If we already mapped annual_spend but later find a better spend column,
        # replace the earlier mapping.
        if canonical_column == "annual_spend":
            preferred_columns = preferred_original_columns["annual_spend"]

            current_original_column = None

            for source_column, mapped_column in rename_mapping.items():
                if mapped_column == "annual_spend":
                    current_original_column = source_column
                    break

            current_rank = (
                preferred_columns.index(
                    normalize_column_name(current_original_column)
                )
                if current_original_column is not None
                   and normalize_column_name(current_original_column) in preferred_columns
                else 999
            )

            new_rank = (
                preferred_columns.index(normalized_column)
                if normalized_column in preferred_columns
                else 999
            )

            if new_rank < current_rank:
                if current_original_column in rename_mapping:
                    del rename_mapping[current_original_column]

                rename_mapping[original_column] = "annual_spend"

                report_rows.append(
                    {
                        "original_column": original_column,
                        "normalized_column": normalized_column,
                        "mapped_column": canonical_column,
                        "column_role": get_column_role(canonical_column),
                        "used_in_analysis": is_column_used_in_analysis(
                            canonical_column
                        ),
                        "mapping_status": "Mapped",
                    }
                )
            else:
                report_rows.append(
                    {
                        "original_column": original_column,
                        "normalized_column": normalized_column,
                        "mapped_column": canonical_column,
                        "column_role": get_column_role(canonical_column),
                        "used_in_analysis": False,
                        "mapping_status": "Duplicate ignored",
                    }
                )

            continue

        report_rows.append(
            {
                "original_column": original_column,
                "normalized_column": normalized_column,
                "mapped_column": canonical_column,
                "column_role": get_column_role(canonical_column),
                "used_in_analysis": False,
                "mapping_status": "Duplicate ignored",
            }
        )

    mapped_data = data.rename(
        columns=rename_mapping
    )

    mapping_report = pd.DataFrame(report_rows)

    return mapped_data, mapping_report