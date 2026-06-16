import pandas as pd

from src.date_utils import (
    add_invoice_period_columns,
    parse_date_column,
    summarize_date_coverage,
)


def test_parse_date_column_converts_valid_dates():
    data = pd.DataFrame(
        {
            "invoice_date": [
                "2025-01-15",
                "2025-02-20",
            ]
        }
    )

    result = parse_date_column(data)

    assert pd.api.types.is_datetime64_any_dtype(
        result["invoice_date"]
    )


def test_parse_date_column_handles_invalid_dates():
    data = pd.DataFrame(
        {
            "invoice_date": [
                "2025-01-15",
                "not a date",
            ]
        }
    )

    result = parse_date_column(data)

    assert result["invoice_date"].notna().sum() == 1
    assert result["invoice_date"].isna().sum() == 1


def test_add_invoice_period_columns_creates_year_quarter_month():
    data = pd.DataFrame(
        {
            "invoice_date": [
                "2025-01-15",
                "2025-04-20",
            ]
        }
    )

    result = add_invoice_period_columns(data)

    assert "invoice_year" in result.columns
    assert "invoice_quarter" in result.columns
    assert "invoice_month" in result.columns

    assert result.loc[0, "invoice_year"] == 2025
    assert result.loc[0, "invoice_quarter"] == "2025Q1"
    assert result.loc[0, "invoice_month"] == "2025-01"

    assert result.loc[1, "invoice_quarter"] == "2025Q2"
    assert result.loc[1, "invoice_month"] == "2025-04"


def test_add_invoice_period_columns_handles_missing_date_column():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight"],
        }
    )

    result = add_invoice_period_columns(data)

    assert "invoice_year" not in result.columns
    assert "invoice_quarter" not in result.columns
    assert "invoice_month" not in result.columns


def test_summarize_date_coverage_without_date_column():
    data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight"],
        }
    )

    summary = summarize_date_coverage(data)

    assert summary["has_date_column"] is False
    assert summary["valid_date_rows"] == 0
    assert summary["invalid_date_rows"] == 0
    assert summary["date_coverage_pct"] == 0.0


def test_summarize_date_coverage_with_mixed_dates():
    data = pd.DataFrame(
        {
            "invoice_date": [
                "2025-01-15",
                "2025-02-20",
                "bad date",
                None,
            ]
        }
    )

    summary = summarize_date_coverage(data)

    assert summary["has_date_column"] is True
    assert summary["valid_date_rows"] == 2
    assert summary["invalid_date_rows"] == 2
    assert summary["date_coverage_pct"] == 50.0
    assert summary["min_date"] == "2025-01-15"
    assert summary["max_date"] == "2025-02-20"