import pandas as pd

from src.column_mapping import (
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