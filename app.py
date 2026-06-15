from pathlib import Path

from src.ingestion import (
    add_internal_supplier_id,
    load_supplier_data,
    summarize_dataset,
    validate_required_columns,
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

    supplier_data = add_internal_supplier_id(
        supplier_data
    )

    missing_columns = validate_required_columns(supplier_data)

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


if __name__ == "__main__":
    main()

