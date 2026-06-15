from pathlib import Path

import pandas as pd


SUPPORTED_FILE_EXTENSIONS = {".csv", ".xlsx", ".xls"}

REQUIRED_COLUMNS = {
    "supplier_id",
    "supplier_name",
    "category",
    "annual_spend",
}


def load_supplier_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load supplier data from a CSV or Excel file.

    Parameters
    ----------
    file_path:
        Path to the input CSV or Excel file.

    Returns
    -------
    pd.DataFrame
        Loaded supplier data.

    Raises
    ------
    FileNotFoundError
        If the requested file does not exist.

    ValueError
        If the file type is unsupported or the file is empty.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"The file does not exist: {file_path}"
        )

    file_extension = file_path.suffix.lower()

    if file_extension not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. "
            "Please provide a CSV or Excel file."
        )

    if file_extension == ".csv":
        data = pd.read_csv(file_path)
    else:
        data = pd.read_excel(file_path)

    if data.empty:
        raise ValueError("The uploaded file contains no data.")

    return data


def validate_required_columns(data: pd.DataFrame) -> list[str]:
    """
    Identify required columns that are missing from the dataset.

    Parameters
    ----------
    data:
        Supplier dataset.

    Returns
    -------
    list[str]
        Missing required column names. Returns an empty list when all
        required columns are present.
    """
    available_columns = set(data.columns)

    missing_columns = sorted(
        REQUIRED_COLUMNS - available_columns
    )

    return missing_columns


def summarize_dataset(data: pd.DataFrame) -> dict[str, int]:
    """
    Return a simple structural summary of the dataset.
    """
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "suppliers": data["supplier_id"].nunique(),
        "categories": data["category"].nunique(),
    }
