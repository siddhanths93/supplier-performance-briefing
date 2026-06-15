from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_SUPPLIERS = 150


CATEGORIES = [
    "IT Hardware",
    "IT Services",
    "Logistics",
    "Facilities",
    "Professional Services",
    "Office Supplies",
    "Packaging",
    "Marketing",
    "Maintenance",
    "Temporary Labor",
]

REGIONS = [
    "North America",
    "Europe",
    "Asia Pacific",
    "Latin America",
]

CONTRACT_STATUSES = [
    "Active",
    "Expired",
    "No Contract",
    "Under Review",
]

CRITICALITY_LEVELS = [
    "Low",
    "Medium",
    "High",
]


def create_supplier_names(number_of_suppliers: int) -> list[str]:
    """
    Create readable synthetic supplier names.

    The names are intentionally generic and do not represent real companies.
    """
    prefixes = [
        "Apex",
        "BlueRiver",
        "ClearPoint",
        "Delta",
        "Evergreen",
        "Fusion",
        "Global",
        "Horizon",
        "Ironwood",
        "Keystone",
        "Lighthouse",
        "Metro",
        "NorthStar",
        "Optima",
        "Pioneer",
    ]

    suffixes = [
        "Solutions",
        "Industries",
        "Group",
        "Services",
        "Partners",
        "Systems",
        "Enterprises",
        "Technologies",
        "Supply",
        "Holdings",
    ]

    supplier_names = []

    for supplier_number in range(1, number_of_suppliers + 1):
        prefix = prefixes[(supplier_number - 1) % len(prefixes)]
        suffix = suffixes[(supplier_number - 1) % len(suffixes)]

        supplier_names.append(
            f"{prefix} {suffix} {supplier_number:03d}"
        )

    return supplier_names


def assign_categories(
    rng: np.random.Generator,
    number_of_suppliers: int,
) -> np.ndarray:
    """
    Assign categories with deliberately uneven supplier counts.

    Office Supplies and Facilities are intentionally more fragmented.
    IT Hardware is intentionally less fragmented.
    """
    category_probabilities = [
        0.04,  # IT Hardware
        0.08,  # IT Services
        0.10,  # Logistics
        0.16,  # Facilities
        0.10,  # Professional Services
        0.18,  # Office Supplies
        0.08,  # Packaging
        0.08,  # Marketing
        0.10,  # Maintenance
        0.08,  # Temporary Labor
    ]

    return rng.choice(
        CATEGORIES,
        size=number_of_suppliers,
        p=category_probabilities,
    )


def generate_base_data() -> pd.DataFrame:
    """
    Generate the initial synthetic supplier dataset.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    supplier_names = create_supplier_names(NUMBER_OF_SUPPLIERS)
    categories = assign_categories(rng, NUMBER_OF_SUPPLIERS)

    annual_spend = rng.lognormal(
        mean=12.0,
        sigma=1.0,
        size=NUMBER_OF_SUPPLIERS,
    )

    annual_spend = np.clip(
        annual_spend,
        10_000,
        5_000_000,
    )

    spend_change = rng.normal(
        loc=0.03,
        scale=0.18,
        size=NUMBER_OF_SUPPLIERS,
    )

    prior_year_spend = annual_spend / (1 + spend_change)

    current_otd = rng.normal(
        loc=91,
        scale=6,
        size=NUMBER_OF_SUPPLIERS,
    )

    prior_otd = current_otd + rng.normal(
        loc=0,
        scale=4,
        size=NUMBER_OF_SUPPLIERS,
    )

    current_defect_rate = rng.gamma(
        shape=2.0,
        scale=1.0,
        size=NUMBER_OF_SUPPLIERS,
    )

    prior_defect_rate = current_defect_rate + rng.normal(
        loc=0,
        scale=0.8,
        size=NUMBER_OF_SUPPLIERS,
    )

    lead_time_days = rng.normal(
        loc=24,
        scale=10,
        size=NUMBER_OF_SUPPLIERS,
    )

    data = pd.DataFrame(
        {
            "supplier_id": [
                f"SUP-{number:04d}"
                for number in range(1, NUMBER_OF_SUPPLIERS + 1)
            ],
            "supplier_name": supplier_names,
            "category": categories,
            "region": rng.choice(
                REGIONS,
                size=NUMBER_OF_SUPPLIERS,
            ),
            "annual_spend": annual_spend,
            "prior_year_spend": prior_year_spend,
            "on_time_delivery_pct": current_otd,
            "prior_year_otd_pct": prior_otd,
            "defect_rate_pct": current_defect_rate,
            "prior_year_defect_rate_pct": prior_defect_rate,
            "lead_time_days": lead_time_days,
            "contract_status": rng.choice(
                CONTRACT_STATUSES,
                size=NUMBER_OF_SUPPLIERS,
                p=[0.58, 0.12, 0.20, 0.10],
            ),
            "supplier_criticality": rng.choice(
                CRITICALITY_LEVELS,
                size=NUMBER_OF_SUPPLIERS,
                p=[0.40, 0.40, 0.20],
            ),
        }
    )

    return data


def apply_business_scenarios(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add deliberate patterns that the analytics should detect later.
    """
    data = data.copy()

    # Scenario 1:
    # Make IT Hardware highly concentrated.
    it_hardware_rows = data["category"] == "IT Hardware"

    if it_hardware_rows.sum() >= 2:
        hardware_indexes = data.index[it_hardware_rows].tolist()

        data.loc[hardware_indexes[0], "annual_spend"] = 5_000_000
        data.loc[hardware_indexes[1], "annual_spend"] = 2_500_000

        for index in hardware_indexes[2:]:
            data.loc[index, "annual_spend"] = min(
                data.loc[index, "annual_spend"],
                250_000,
            )

    # Scenario 2:
    # Create a high-spend Logistics supplier with deteriorating delivery.
    logistics_indexes = data.index[
        data["category"] == "Logistics"
    ].tolist()

    if logistics_indexes:
        index = logistics_indexes[0]

        data.loc[index, "supplier_name"] = (
            "Apex Freight Solutions"
        )
        data.loc[index, "annual_spend"] = 4_200_000
        data.loc[index, "prior_year_spend"] = 3_900_000
        data.loc[index, "on_time_delivery_pct"] = 81.0
        data.loc[index, "prior_year_otd_pct"] = 94.0
        data.loc[index, "defect_rate_pct"] = 4.8
        data.loc[index, "prior_year_defect_rate_pct"] = 2.1
        data.loc[index, "supplier_criticality"] = "High"
        data.loc[index, "contract_status"] = "Active"

    # Scenario 3:
    # Create a supplier with worsening quality.
    facilities_indexes = data.index[
        data["category"] == "Facilities"
    ].tolist()

    if facilities_indexes:
        index = facilities_indexes[0]

        data.loc[index, "supplier_name"] = (
            "Metro Facility Services"
        )
        data.loc[index, "annual_spend"] = 1_750_000
        data.loc[index, "defect_rate_pct"] = 7.2
        data.loc[index, "prior_year_defect_rate_pct"] = 2.4
        data.loc[index, "on_time_delivery_pct"] = 88.0
        data.loc[index, "prior_year_otd_pct"] = 91.0
        data.loc[index, "supplier_criticality"] = "High"

    # Scenario 4:
    # Create a supplier with improving performance.
    packaging_indexes = data.index[
        data["category"] == "Packaging"
    ].tolist()

    if packaging_indexes:
        index = packaging_indexes[0]

        data.loc[index, "supplier_name"] = (
            "Pioneer Packaging Group"
        )
        data.loc[index, "annual_spend"] = 1_100_000
        data.loc[index, "on_time_delivery_pct"] = 97.0
        data.loc[index, "prior_year_otd_pct"] = 86.0
        data.loc[index, "defect_rate_pct"] = 1.1
        data.loc[index, "prior_year_defect_rate_pct"] = 4.5

    # Scenario 5:
    # Create low-value tail suppliers in Office Supplies.
    office_indexes = data.index[
        data["category"] == "Office Supplies"
    ].tolist()

    for index in office_indexes:
        data.loc[index, "annual_spend"] = min(
            data.loc[index, "annual_spend"],
            90_000,
        )

    # Scenario 6:
    # Add a few inconsistent supplier names.
    if len(data) >= 10:
        data.loc[2, "supplier_name"] = "Global Systems LLC"
        data.loc[5, "supplier_name"] = "GLOBAL SYSTEMS, L.L.C."
        data.loc[8, "supplier_name"] = "Global System LLC"

    # Scenario 7:
    # Add missing performance values.
    missing_otd_indexes = [12, 37, 89]
    missing_quality_indexes = [21, 64, 110]

    for index in missing_otd_indexes:
        if index < len(data):
            data.loc[index, "on_time_delivery_pct"] = np.nan
            data.loc[index, "prior_year_otd_pct"] = np.nan

    for index in missing_quality_indexes:
        if index < len(data):
            data.loc[index, "defect_rate_pct"] = np.nan
            data.loc[index, "prior_year_defect_rate_pct"] = np.nan

    return data


def clean_and_format_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply sensible numeric limits and formatting.
    """
    data = data.copy()

    data["annual_spend"] = data["annual_spend"].round(2)
    data["prior_year_spend"] = (
        data["prior_year_spend"]
        .clip(lower=1_000)
        .round(2)
    )

    data["on_time_delivery_pct"] = (
        data["on_time_delivery_pct"]
        .clip(lower=50, upper=100)
        .round(1)
    )

    data["prior_year_otd_pct"] = (
        data["prior_year_otd_pct"]
        .clip(lower=50, upper=100)
        .round(1)
    )

    data["defect_rate_pct"] = (
        data["defect_rate_pct"]
        .clip(lower=0, upper=15)
        .round(1)
    )

    data["prior_year_defect_rate_pct"] = (
        data["prior_year_defect_rate_pct"]
        .clip(lower=0, upper=15)
        .round(1)
    )

    data["lead_time_days"] = (
        data["lead_time_days"]
        .clip(lower=2, upper=90)
        .round()
        .astype(int)
    )

    return data


def save_dataset(data: pd.DataFrame) -> Path:
    """
    Save the synthetic dataset into the data/sample folder.
    """
    project_root = Path(__file__).resolve().parent.parent
    output_directory = project_root / "data" / "sample"
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / "synthetic_supplier_performance.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def main() -> None:
    """
    Generate, format, save, and summarize the synthetic dataset.
    """
    supplier_data = generate_base_data()
    supplier_data = apply_business_scenarios(supplier_data)
    supplier_data = clean_and_format_data(supplier_data)

    output_path = save_dataset(supplier_data)

    print("Synthetic supplier dataset created successfully.")
    print(f"File location: {output_path}")
    print(f"Number of rows: {len(supplier_data)}")
    print(f"Number of categories: {supplier_data['category'].nunique()}")
    print(
        "Total annual spend: "
        f"${supplier_data['annual_spend'].sum():,.2f}"
    )


if __name__ == "__main__":
    main()