from pathlib import Path

import pandas as pd
import pytest

from src.ingestion import (
    add_internal_supplier_id,
    load_supplier_data,
    validate_required_columns,
)


def test_load_supplier_data_loads_csv_file(tmp_path):
    test_file = tmp_path / "supplier_data.csv"

    test_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [1200000],
        }
    )

    test_data.to_csv(test_file, index=False)

    loaded_data = load_supplier_data(test_file)

    assert len(loaded_data) == 1
    assert loaded_data.loc[0, "supplier_name"] == "Apex Freight Solutions"
    assert loaded_data.loc[0, "category"] == "Logistics"
    assert loaded_data.loc[0, "annual_spend"] == 1200000


def test_load_supplier_data_raises_error_for_missing_file():
    missing_file = Path("data/does_not_exist.csv")

    with pytest.raises(FileNotFoundError):
        load_supplier_data(missing_file)


def test_load_supplier_data_raises_error_for_unsupported_file_type(tmp_path):
    test_file = tmp_path / "supplier_data.txt"
    test_file.write_text("not a valid supplier file")

    with pytest.raises(ValueError):
        load_supplier_data(test_file)


def test_validate_required_columns_passes_when_columns_exist():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [1200000],
        }
    )

    missing_columns = validate_required_columns(supplier_data)

    assert missing_columns == []


def test_validate_required_columns_detects_missing_annual_spend():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
        }
    )

    missing_columns = validate_required_columns(supplier_data)

    assert missing_columns == ["annual_spend"]


def test_add_internal_supplier_id_when_missing():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": [
                "Apex Freight Solutions",
                "Metro Facility Services",
            ],
            "category": [
                "Logistics",
                "Facilities",
            ],
            "annual_spend": [
                1200000,
                800000,
            ],
        }
    )

    updated_data = add_internal_supplier_id(supplier_data)

    assert "supplier_id" in updated_data.columns
    assert updated_data.loc[0, "supplier_id"] == "SUP-0001"
    assert updated_data.loc[1, "supplier_id"] == "SUP-0002"


def test_add_internal_supplier_id_preserves_existing_supplier_id():
    supplier_data = pd.DataFrame(
        {
            "supplier_id": ["V001"],
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [1200000],
        }
    )

    updated_data = add_internal_supplier_id(supplier_data)

    assert updated_data.loc[0, "supplier_id"] == "V001"