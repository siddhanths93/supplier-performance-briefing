import pandas as pd
import pytest

from src.aggregation import (
    aggregate_supplier_category_data,
    create_aggregation_summary,
    first_non_null,
    get_grouping_columns,
    should_aggregate_supplier_data,
)


def test_get_grouping_columns_without_time_grain():
    assert get_grouping_columns() == [
        "supplier_name",
        "category",
    ]


def test_get_grouping_columns_with_year_grain():
    assert get_grouping_columns("year") == [
        "supplier_name",
        "category",
        "invoice_year",
    ]


def test_get_grouping_columns_with_quarter_grain():
    assert get_grouping_columns("quarter") == [
        "supplier_name",
        "category",
        "invoice_quarter",
    ]


def test_get_grouping_columns_with_month_grain():
    assert get_grouping_columns("month") == [
        "supplier_name",
        "category",
        "invoice_month",
    ]


def test_get_grouping_columns_rejects_invalid_grain():
    with pytest.raises(ValueError):
        get_grouping_columns("week")


def test_first_non_null_returns_first_available_value():
    series = pd.Series(
        [
            None,
            pd.NA,
            "Logistics",
            "IT",
        ]
    )

    assert first_non_null(series) == "Logistics"


def test_should_aggregate_supplier_data_returns_true_for_duplicates():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight",
                "Apex Freight",
            ],
            "category": [
                "Freight & Parcel",
                "Freight & Parcel",
            ],
            "annual_spend": [
                100,
                200,
            ],
        }
    )

    assert should_aggregate_supplier_data(data) is True


def test_should_aggregate_supplier_data_respects_year_grain():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight",
                "Apex Freight",
            ],
            "category": [
                "Freight & Parcel",
                "Freight & Parcel",
            ],
            "invoice_year": [
                2024,
                2025,
            ],
            "annual_spend": [
                100,
                200,
            ],
        }
    )

    assert should_aggregate_supplier_data(data) is True
    assert should_aggregate_supplier_data(data, time_grain="year") is False


def test_aggregate_supplier_category_data_sums_spend_without_time_grain():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight",
                "Apex Freight",
                "Bright Software",
            ],
            "category": [
                "Freight & Parcel",
                "Freight & Parcel",
                "Software",
            ],
            "annual_spend": [
                100,
                200,
                500,
            ],
        }
    )

    aggregated_data = aggregate_supplier_category_data(data)

    apex_row = aggregated_data[
        aggregated_data["supplier_name"] == "Apex Freight"
    ].iloc[0]

    assert apex_row["annual_spend"] == 300
    assert apex_row["transaction_count"] == 2


def test_aggregate_supplier_category_data_sums_spend_by_year():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight",
                "Apex Freight",
                "Apex Freight",
            ],
            "category": [
                "Freight & Parcel",
                "Freight & Parcel",
                "Freight & Parcel",
            ],
            "invoice_year": [
                2024,
                2025,
                2025,
            ],
            "annual_spend": [
                100,
                200,
                300,
            ],
        }
    )

    aggregated_data = aggregate_supplier_category_data(
        data,
        time_grain="year",
    )

    assert len(aggregated_data) == 2

    year_2025_row = aggregated_data[
        aggregated_data["invoice_year"] == 2025
    ].iloc[0]

    assert year_2025_row["annual_spend"] == 500
    assert year_2025_row["transaction_count"] == 2


def test_aggregate_supplier_category_data_averages_performance_fields():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight",
                "Apex Freight",
            ],
            "category": [
                "Freight & Parcel",
                "Freight & Parcel",
            ],
            "annual_spend": [
                100,
                200,
            ],
            "on_time_delivery_pct": [
                90,
                80,
            ],
            "defect_rate_pct": [
                2,
                4,
            ],
        }
    )

    aggregated_data = aggregate_supplier_category_data(data)

    row = aggregated_data.iloc[0]

    assert row["annual_spend"] == 300
    assert row["on_time_delivery_pct"] == 85
    assert row["defect_rate_pct"] == 3


def test_aggregate_supplier_category_data_keeps_first_context_value():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight",
                "Apex Freight",
            ],
            "category": [
                "Freight & Parcel",
                "Freight & Parcel",
            ],
            "annual_spend": [
                100,
                200,
            ],
            "region": [
                None,
                "North America",
            ],
        }
    )

    aggregated_data = aggregate_supplier_category_data(data)

    row = aggregated_data.iloc[0]

    assert row["region"] == "North America"


def test_create_aggregation_summary_counts_reduction():
    before_data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight",
                "Apex Freight",
                "Bright Software",
            ],
            "category": [
                "Freight & Parcel",
                "Freight & Parcel",
                "Software",
            ],
            "annual_spend": [
                100,
                200,
                500,
            ],
        }
    )

    after_data = aggregate_supplier_category_data(
        before_data
    )

    summary = create_aggregation_summary(
        before_data=before_data,
        after_data=after_data,
        was_aggregated=True,
        time_grain=None,
    )

    assert summary["was_aggregated"] is True
    assert summary["time_grain"] == "none"
    assert summary["input_rows"] == 3
    assert summary["output_rows"] == 2
    assert summary["rows_reduced"] == 1
    assert summary["input_supplier_count"] == 2
    assert summary["output_supplier_count"] == 2