from pathlib import Path

from src.ingestion import (
    add_internal_supplier_id,
    load_supplier_data,
    summarize_dataset,
    validate_required_columns,
)

from src.validation import (
    calculate_missing_values,
    create_data_quality_summary,
)

from src.cleaning import clean_supplier_data

from src.supplier_metrics import calculate_supplier_metrics

from src.category_metrics import (
    calculate_category_metrics,
)

def main() -> None:
    """
    Load and validate the synthetic supplier dataset.
    """
    project_root = Path(__file__).resolve().parent

    sample_file = (
        project_root
        / "data"
        / "sample"
        / "synthetic_supplier_performance.csv"
    )

    supplier_data = load_supplier_data(sample_file)

    supplier_data = clean_supplier_data(
        supplier_data
    )

    supplier_data = add_internal_supplier_id(
        supplier_data
    )

    supplier_data = calculate_supplier_metrics(
        supplier_data
    )

    category_metrics = calculate_category_metrics(
        supplier_data
    )
    missing_columns = validate_required_columns(supplier_data)

    data_quality_summary = create_data_quality_summary(
        supplier_data
    )

    missing_value_summary = calculate_missing_values(
        supplier_data
    )

    if missing_columns:
        print("Dataset validation failed.")
        print(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )
        return

    dataset_summary = summarize_dataset(supplier_data)

    print("Supplier dataset loaded successfully.")
    print(f"Rows: {dataset_summary['rows']}")
    print(f"Columns: {dataset_summary['columns']}")
    print(f"Suppliers: {dataset_summary['suppliers']}")
    print(f"Categories: {dataset_summary['categories']}")
    print("\nData quality summary")
    print("--------------------")
    print(
        f"Missing cells: "
        f"{data_quality_summary['missing_cells']}"
    )
    print(
        f"Missing cell percentage: "
        f"{data_quality_summary['missing_cell_pct']}%"
    )
    print(
        f"Exact duplicate rows: "
        f"{data_quality_summary['exact_duplicate_rows']}"
    )
    print(
        "Duplicate supplier-name rows: "
        f"{data_quality_summary['duplicate_supplier_name_rows']}"
    )
    print(
        f"Invalid spend rows: "
        f"{data_quality_summary['invalid_spend_rows']}"
    )
    print(
        "Invalid percentage values: "
        f"{data_quality_summary['invalid_percentage_values']}"
    )

    print("\nColumns with missing values")
    print("---------------------------")

    columns_with_missing_values = missing_value_summary[
        missing_value_summary["missing_count"] > 0
        ]

    if columns_with_missing_values.empty:
        print("No missing values found.")
    else:
        print(
            columns_with_missing_values.to_string(
                index=False
            )
        )

    print("\nSupplier metrics sample")
    print("-----------------------")

    metric_columns = [
        "supplier_name",
        "category",
        "annual_spend",
        "spend_change_pct",
        "category_spend_share_pct",
        "otd_change_pct_points",
        "defect_rate_change_pct_points",
        "delivery_deterioration_flag",
        "quality_deterioration_flag",
        "high_spend_exposure_flag",
    ]

    print(
        supplier_data[metric_columns]
        .sort_values(
            by="annual_spend",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print("\nCategory concentration and fragmentation")
    print("----------------------------------------")

    category_columns = [
        "category",
        "total_category_spend",
        "supplier_count",
        "top_supplier_share_pct",
        "top_3_supplier_share_pct",
        "suppliers_to_80_pct_spend",
        "tail_supplier_pct",
        "tail_spend_pct",
        "hhi",
        "concentration_risk_flag",
        "fragmentation_review_flag",
        "tail_spend_review_flag",
    ]

    print(
        category_metrics[category_columns]
        .sort_values(
            by="total_category_spend",
            ascending=False,
        )
        .to_string(index=False)
    )

if __name__ == "__main__":
    main()

