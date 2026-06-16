import pandas as pd


DATE_COLUMNS = [
    "invoice_date",
]


def parse_date_column(
    data: pd.DataFrame,
    date_column: str = "invoice_date",
) -> pd.DataFrame:
    """
    Parse a date column into pandas datetime.

    Invalid or blank dates become NaT instead of breaking the app.
    """
    data = data.copy()

    if date_column not in data.columns:
        return data

    data[date_column] = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    return data


def add_invoice_period_columns(
    data: pd.DataFrame,
    date_column: str = "invoice_date",
) -> pd.DataFrame:
    """
    Add year, quarter, and month fields from invoice_date.
    """
    data = data.copy()

    if date_column not in data.columns:
        return data

    data = parse_date_column(
        data,
        date_column=date_column,
    )

    data["invoice_year"] = data[date_column].dt.year.astype("Int64")

    data["invoice_quarter"] = data[date_column].dt.to_period(
        "Q"
    ).astype(str)

    data.loc[
        data[date_column].isna(),
        "invoice_quarter",
    ] = pd.NA

    data["invoice_month"] = data[date_column].dt.to_period(
        "M"
    ).astype(str)

    data.loc[
        data[date_column].isna(),
        "invoice_month",
    ] = pd.NA

    return data


def summarize_date_coverage(
    data: pd.DataFrame,
    date_column: str = "invoice_date",
) -> dict:
    """
    Summarize whether uploaded data has usable date coverage.
    """
    if date_column not in data.columns:
        return {
            "has_date_column": False,
            "valid_date_rows": 0,
            "invalid_date_rows": 0,
            "date_coverage_pct": 0.0,
            "min_date": None,
            "max_date": None,
        }

    parsed_dates = pd.to_datetime(
        data[date_column],
        errors="coerce",
    )

    total_rows = len(data)
    valid_date_rows = int(parsed_dates.notna().sum())
    invalid_date_rows = total_rows - valid_date_rows

    date_coverage_pct = (
        valid_date_rows / total_rows * 100
        if total_rows > 0
        else 0.0
    )

    min_date = (
        parsed_dates.min().date().isoformat()
        if valid_date_rows > 0
        else None
    )

    max_date = (
        parsed_dates.max().date().isoformat()
        if valid_date_rows > 0
        else None
    )

    return {
        "has_date_column": True,
        "valid_date_rows": valid_date_rows,
        "invalid_date_rows": invalid_date_rows,
        "date_coverage_pct": round(date_coverage_pct, 1),
        "min_date": min_date,
        "max_date": max_date,
    }