import pandas as pd

from src.column_mapping import (
    get_column_role,
    is_column_used_in_analysis,
    map_columns,
    normalize_column_name,
)


def test_normalize_column_name_handles_spacing_and_symbols():
    assert normalize_column_name(" Vendor_Name ") == "vendor name"
    assert normalize_column_name("Invoice-Amount") == "invoice amount"
    assert normalize_column_name("OTD %") == "otd"


def test_map_columns_maps_common_supplier_file_columns():
    raw_data = pd.DataFrame(
        {
            "Vendor Name": ["Apex Freight Solutions"],
            "Commodity Group": ["Logistics"],
            "Invoice Amount": [120000],
            "Invoice Description": ["Freight shipment"],
        }
    )

    mapped_data, mapping_report = map_columns(raw_data)

    assert "supplier_name" in mapped_data.columns
    assert "category" in mapped_data.columns
    assert "annual_spend" in mapped_data.columns
    assert "description" in mapped_data.columns

    assert len(mapping_report) == 4


def test_map_columns_leaves_unmapped_columns_unchanged():
    raw_data = pd.DataFrame(
        {
            "Vendor Name": ["Apex Freight Solutions"],
            "Random Internal Field": ["ABC123"],
        }
    )

    mapped_data, mapping_report = map_columns(raw_data)

    assert "supplier_name" in mapped_data.columns
    assert "Random Internal Field" in mapped_data.columns

    unmapped_rows = mapping_report[
        mapping_report["mapping_status"] == "Unmapped"
    ]

    assert len(unmapped_rows) == 1
    assert unmapped_rows.iloc[0]["column_role"] == "Unmapped"
    assert unmapped_rows.iloc[0]["used_in_analysis"] == False


def test_map_columns_ignores_duplicate_aliases_for_same_field():
    raw_data = pd.DataFrame(
        {
            "Vendor Name": ["Apex Freight Solutions"],
            "Supplier Name": ["Apex Freight Solutions LLC"],
            "Invoice Amount": [120000],
        }
    )

    mapped_data, mapping_report = map_columns(raw_data)

    supplier_name_count = list(
        mapped_data.columns
    ).count("supplier_name")

    assert supplier_name_count == 1

    duplicate_rows = mapping_report[
        mapping_report["mapping_status"]
        == "Duplicate ignored"
    ]

    assert len(duplicate_rows) == 1


def test_map_columns_identifies_context_columns():
    raw_data = pd.DataFrame(
        {
            "Vendor Name": ["Apex Freight Solutions"],
            "Invoice Amount": [120000],
            "Cost Center": ["CC100"],
            "Business Unit": ["North America"],
            "Payment Terms": ["Net 60"],
            "PO Number": ["PO123"],
        }
    )

    mapped_data, mapping_report = map_columns(raw_data)

    assert "cost_center" in mapped_data.columns
    assert "business_unit" in mapped_data.columns
    assert "payment_terms" in mapped_data.columns
    assert "po_number" in mapped_data.columns

    context_rows = mapping_report[
        mapping_report["column_role"] == "Context"
    ]

    assert len(context_rows) == 4


def test_get_column_role_returns_expected_roles():
    assert get_column_role("supplier_name") == "Required"
    assert get_column_role("description") == "Classification"
    assert get_column_role("on_time_delivery_pct") == "Full scoring"
    assert get_column_role("cost_center") == "Context"
    assert get_column_role(None) == "Unmapped"


def test_is_column_used_in_analysis_identifies_used_columns():
    assert is_column_used_in_analysis("supplier_name") == True
    assert is_column_used_in_analysis("annual_spend") == True
    assert is_column_used_in_analysis("cost_center") == False
    assert is_column_used_in_analysis(None) == False