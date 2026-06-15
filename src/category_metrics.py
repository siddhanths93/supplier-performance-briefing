import numpy as np
import pandas as pd


def calculate_category_summary(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate core category-level spend and supplier metrics.
    """
    required_columns = {
        "category",
        "supplier_id",
        "annual_spend",
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing required columns for category analysis: "
            + ", ".join(sorted(missing_columns))
        )

    category_summary = (
        data.groupby("category", dropna=False)
        .agg(
            total_category_spend=(
                "annual_spend",
                "sum",
            ),
            supplier_count=(
                "supplier_id",
                "nunique",
            ),
            average_spend_per_supplier=(
                "annual_spend",
                "mean",
            ),
            median_spend_per_supplier=(
                "annual_spend",
                "median",
            ),
        )
        .reset_index()
    )

    return category_summary


def calculate_supplier_concentration(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate supplier concentration metrics by category.

    Metrics:
    - top supplier share
    - top three supplier share
    - top five supplier share
    - suppliers required to reach 80% of category spend
    - Herfindahl-Hirschman Index
    """
    concentration_results = []

    for category, category_data in data.groupby(
        "category",
        dropna=False,
    ):
        supplier_spend = (
            category_data.groupby("supplier_id")[
                "annual_spend"
            ]
            .sum()
            .sort_values(ascending=False)
        )

        total_spend = supplier_spend.sum()

        if total_spend <= 0:
            supplier_shares = pd.Series(
                dtype="float64"
            )
        else:
            supplier_shares = (
                supplier_spend / total_spend
            )

        cumulative_shares = supplier_shares.cumsum()

        if supplier_shares.empty:
            suppliers_to_80_pct = np.nan
        else:
            suppliers_to_80_pct = int(
                (cumulative_shares < 0.80).sum() + 1
            )

        hhi = float(
            ((supplier_shares * 100) ** 2).sum()
        )

        concentration_results.append(
            {
                "category": category,
                "top_supplier_share_pct": round(
                    supplier_shares.head(1).sum() * 100,
                    2,
                ),
                "top_3_supplier_share_pct": round(
                    supplier_shares.head(3).sum() * 100,
                    2,
                ),
                "top_5_supplier_share_pct": round(
                    supplier_shares.head(5).sum() * 100,
                    2,
                ),
                "suppliers_to_80_pct_spend": (
                    suppliers_to_80_pct
                ),
                "hhi": round(hhi, 2),
            }
        )

    return pd.DataFrame(concentration_results)


def calculate_tail_supplier_metrics(
    data: pd.DataFrame,
    tail_spend_threshold: float = 100_000,
) -> pd.DataFrame:
    """
    Calculate low-spend supplier metrics by category.

    A tail supplier is currently defined as a supplier with
    annual spend below the chosen threshold.
    """
    tail_results = []

    for category, category_data in data.groupby(
        "category",
        dropna=False,
    ):
        supplier_spend = (
            category_data.groupby("supplier_id")[
                "annual_spend"
            ]
            .sum()
        )

        total_category_spend = supplier_spend.sum()

        tail_suppliers = supplier_spend[
            supplier_spend < tail_spend_threshold
        ]

        tail_supplier_count = len(tail_suppliers)
        total_supplier_count = len(supplier_spend)
        tail_spend = tail_suppliers.sum()

        tail_supplier_pct = (
            tail_supplier_count
            / total_supplier_count
            * 100
            if total_supplier_count > 0
            else 0.0
        )

        tail_spend_pct = (
            tail_spend
            / total_category_spend
            * 100
            if total_category_spend > 0
            else 0.0
        )

        tail_results.append(
            {
                "category": category,
                "tail_supplier_count": (
                    tail_supplier_count
                ),
                "tail_supplier_pct": round(
                    tail_supplier_pct,
                    2,
                ),
                "tail_spend": round(
                    tail_spend,
                    2,
                ),
                "tail_spend_pct": round(
                    tail_spend_pct,
                    2,
                ),
            }
        )

    return pd.DataFrame(tail_results)


def add_category_review_flags(
    category_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add transparent category review flags.

    These flags indicate where further review may be useful.
    They do not represent guaranteed savings.
    """
    category_metrics = category_metrics.copy()

    category_metrics[
        "concentration_risk_flag"
    ] = (
        category_metrics[
            "top_supplier_share_pct"
        ]
        >= 60
    )

    category_metrics[
        "fragmentation_review_flag"
    ] = (
        (
            category_metrics["supplier_count"]
            >= 15
        )
        & (
            category_metrics[
                "top_3_supplier_share_pct"
            ]
            <= 50
        )
    )

    category_metrics[
        "tail_spend_review_flag"
    ] = (
        (
            category_metrics[
                "tail_supplier_pct"
            ]
            >= 50
        )
        & (
            category_metrics[
                "tail_spend_pct"
            ]
            >= 10
        )
    )

    return category_metrics


def calculate_category_metrics(
    data: pd.DataFrame,
    tail_spend_threshold: float = 100_000,
) -> pd.DataFrame:
    """
    Run the complete category-level metrics pipeline.
    """
    category_summary = calculate_category_summary(
        data
    )

    concentration_metrics = (
        calculate_supplier_concentration(data)
    )

    tail_metrics = calculate_tail_supplier_metrics(
        data,
        tail_spend_threshold=tail_spend_threshold,
    )

    category_metrics = category_summary.merge(
        concentration_metrics,
        on="category",
        how="left",
    )

    category_metrics = category_metrics.merge(
        tail_metrics,
        on="category",
        how="left",
    )

    category_metrics = add_category_review_flags(
        category_metrics
    )

    return category_metrics