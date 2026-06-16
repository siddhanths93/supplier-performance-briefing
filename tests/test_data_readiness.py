import pandas as pd

from src.data_readiness import (
    create_data_readiness_report,
    detect_input_file_type,
    evaluate_minimum_required_columns,
)


def test_minimum_required_ready_with_category():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": [120000],
            "category": ["Logistics"],
        }
    )

    ready, missing_columns = evaluate_minimum_required_columns(
        data
    )

    assert ready is True
    assert missing_columns == []


def test_minimum_required_ready_with_description():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": [120000],
            "description": ["Freight shipment"],
        }
    )

    ready, missing_columns = evaluate_minimum_required_columns(
        data
    )

    assert ready is True
    assert missing_columns == []


def test_minimum_required_missing_category_and_description():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": [120000],
        }
    )

    ready, missing_columns = evaluate_minimum_required_columns(
        data
    )

    assert ready is False
    assert "category or description" in missing_columns


def test_detect_transaction_level_file():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": [120000],
            "description": ["Freight shipment"],
            "invoice_date": ["2025-01-01"],
            "po_number": ["PO123"],
        }
    )

    file_type = detect_input_file_type(data)

    assert file_type == "Transaction-level spend file"


def test_detect_multi_row_supplier_file():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight Solutions",
                "Apex Freight Solutions",
            ],
            "annual_spend": [
                120000,
                80000,
            ],
            "category": [
                "Logistics",
                "Logistics",
            ],
        }
    )

    file_type = detect_input_file_type(data)

    assert file_type == "Likely multi-row supplier spend file"


def test_create_data_readiness_report_ready_with_limitations():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": [120000],
            "category": ["Logistics"],
        }
    )

    report = create_data_readiness_report(data)

    assert report["analysis_status"] == "Ready with Limitations"
    assert report["minimum_required_ready"] is True
    assert report["input_file_type"] == "Supplier-level performance file"


def test_create_data_readiness_report_not_ready():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
        }
    )

    report = create_data_readiness_report(data)

    assert report["analysis_status"] == "Not Ready"
    assert report["minimum_required_ready"] is False
    assert "annual_spend" in report["missing_required_columns"]


def test_create_data_readiness_report_summarizes_mapping_report():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": [120000],
            "category": ["Logistics"],
            "cost_center": ["CC100"],
            "random_field": ["ABC"],
        }
    )

    mapping_report = pd.DataFrame(
        {
            "original_column": [
                "Vendor Name",
                "Invoice Amount",
                "Cost Center",
                "Random Field",
            ],
            "mapped_column": [
                "supplier_name",
                "annual_spend",
                "cost_center",
                "",
            ],
            "column_role": [
                "Required",
                "Required",
                "Context",
                "Unmapped",
            ],
            "used_in_analysis": [
                True,
                True,
                False,
                False,
            ],
            "mapping_status": [
                "Mapped",
                "Mapped",
                "Mapped",
                "Unmapped",
            ],
        }
    )

    report = create_data_readiness_report(
        data,
        mapping_report=mapping_report,
    )

    assert report["mapped_column_count"] == 3
    assert report["context_column_count"] == 1
    assert report["unmapped_column_count"] == 1
    assert report["used_in_analysis_count"] == 2