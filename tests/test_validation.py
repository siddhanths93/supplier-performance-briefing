import pandas as pd

from src.validation import (
    calculate_missing_values,
    count_exact_duplicate_rows,
    count_invalid_percentage_values,
    count_invalid_spend_values,
    create_data_quality_summary,
)


def test_calculate_missing_values_counts_missing_values():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight Solutions",
                None,
            ],
            "category": [
                "Logistics",
                "Facilities",
            ],
            "annual_spend": [
                1200000,
                None,
            ],
        }
    )

    missing_summary = calculate_missing_values(supplier_data)

    supplier_name_missing = missing_summary[
        missing_summary["column"] == "supplier_name"
    ].iloc[0]

    annual_spend_missing = missing_summary[
        missing_summary["column"] == "annual_spend"
    ].iloc[0]

    assert supplier_name_missing["missing_count"] == 1
    assert annual_spend_missing["missing_count"] == 1


def test_count_exact_duplicate_rows_detects_duplicates():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight Solutions",
                "Apex Freight Solutions",
                "Metro Facility Services",
            ],
            "category": [
                "Logistics",
                "Logistics",
                "Facilities",
            ],
            "annual_spend": [
                1200000,
                1200000,
                800000,
            ],
        }
    )

    duplicate_count = count_exact_duplicate_rows(
        supplier_data
    )

    assert duplicate_count == 1


def test_count_invalid_spend_values_detects_missing_zero_and_negative():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight Solutions",
                "Metro Facility Services",
                "Pioneer Packaging Group",
                "Global Systems LLC",
            ],
            "annual_spend": [
                1200000,
                0,
                -100,
                None,
            ],
        }
    )

    invalid_spend_count = count_invalid_spend_values(
        supplier_data
    )

    assert invalid_spend_count == 3


def test_count_invalid_percentage_values_detects_out_of_range_values():
    supplier_data = pd.DataFrame(
        {
            "on_time_delivery_pct": [
                95,
                105,
                -5,
            ],
            "prior_year_otd_pct": [
                90,
                80,
                70,
            ],
            "defect_rate_pct": [
                2,
                101,
                5,
            ],
            "prior_year_defect_rate_pct": [
                1,
                2,
                -1,
            ],
        }
    )

    invalid_percentage_count = (
        count_invalid_percentage_values(
            supplier_data
        )
    )

    assert invalid_percentage_count == 4


def test_create_data_quality_summary_returns_expected_counts():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight Solutions",
                "Apex Freight Solutions",
                "Metro Facility Services",
            ],
            "category": [
                "Logistics",
                "Logistics",
                "Facilities",
            ],
            "annual_spend": [
                1200000,
                1200000,
                -50,
            ],
            "on_time_delivery_pct": [
                95,
                95,
                110,
            ],
            "prior_year_otd_pct": [
                90,
                90,
                80,
            ],
            "defect_rate_pct": [
                2,
                2,
                None,
            ],
            "prior_year_defect_rate_pct": [
                1,
                1,
                2,
            ],
        }
    )

    summary = create_data_quality_summary(
        supplier_data
    )

    assert summary["rows"] == 3
    assert summary["columns"] == 7
    assert summary["missing_cells"] == 1
    assert summary["exact_duplicate_rows"] == 1
    assert summary["invalid_spend_rows"] == 1
    assert summary["invalid_percentage_values"] == 1