import pandas as pd

from src.schema import ensure_optional_analysis_columns


def test_ensure_optional_analysis_columns_adds_missing_columns():
    data = pd.DataFrame(
        {
            "supplier_name": ["DHL Express"],
            "category": ["Freight & Parcel"],
            "annual_spend": [1000],
        }
    )

    result = ensure_optional_analysis_columns(data)

    assert "prior_year_spend" in result.columns
    assert "on_time_delivery_pct" in result.columns
    assert "prior_year_otd_pct" in result.columns
    assert "defect_rate_pct" in result.columns
    assert "prior_year_defect_rate_pct" in result.columns
    assert "supplier_criticality" in result.columns


def test_ensure_optional_analysis_columns_preserves_existing_values():
    data = pd.DataFrame(
        {
            "supplier_name": ["DHL Express"],
            "category": ["Freight & Parcel"],
            "annual_spend": [1000],
            "prior_year_spend": [800],
        }
    )

    result = ensure_optional_analysis_columns(data)

    assert result.loc[0, "prior_year_spend"] == 800