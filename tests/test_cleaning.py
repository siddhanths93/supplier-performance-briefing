import pandas as pd

from src.cleaning import clean_supplier_data


def test_clean_supplier_data_standardizes_column_names():
    raw_data = pd.DataFrame(
        {
            "Supplier Name": [" Apex Freight Solutions "],
            "Annual Spend": ["$1,200,000"],
            "Category": ["logistics"],
            "Region": ["NA"],
        }
    )

    cleaned_data = clean_supplier_data(raw_data)

    assert "supplier_name" in cleaned_data.columns
    assert "annual_spend" in cleaned_data.columns
    assert "category" in cleaned_data.columns
    assert "region" in cleaned_data.columns


def test_clean_supplier_data_converts_numeric_columns():
    raw_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": ["$1,200,000"],
            "category": ["Logistics"],
        }
    )

    cleaned_data = clean_supplier_data(raw_data)

    assert cleaned_data.loc[0, "annual_spend"] == 1200000


def test_clean_supplier_data_standardizes_region_values():
    raw_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "annual_spend": [1200000],
            "category": ["Logistics"],
            "region": ["NA"],
        }
    )

    cleaned_data = clean_supplier_data(raw_data)

    assert cleaned_data.loc[0, "region"] == "North America"