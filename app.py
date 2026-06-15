from pathlib import Path

import pandas as pd
import streamlit as st

from src.category_metrics import calculate_category_metrics
from src.cleaning import clean_supplier_data
from src.ingestion import (
    add_internal_supplier_id,
    load_supplier_data,
    validate_required_columns,
)
from src.recommendations import (
    create_category_findings,
    create_supplier_findings,
)
from src.scoring import add_supplier_attention_scores
from src.supplier_metrics import calculate_supplier_metrics
from src.validation import create_data_quality_summary


PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "synthetic_supplier_performance.csv"
)


def prepare_supplier_data(
    raw_data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Clean, validate, score, and summarize supplier data.

    Returns
    -------
    tuple
        supplier_data
        category_metrics
        supplier_findings
        category_findings
    """
    supplier_data = clean_supplier_data(raw_data)

    missing_columns = validate_required_columns(
        supplier_data
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    supplier_data = add_internal_supplier_id(
        supplier_data
    )

    supplier_data = calculate_supplier_metrics(
        supplier_data
    )

    supplier_data = add_supplier_attention_scores(
        supplier_data
    )

    category_metrics = calculate_category_metrics(
        supplier_data
    )

    supplier_findings = create_supplier_findings(
        supplier_data,
        top_n=10,
    )

    category_findings = create_category_findings(
        category_metrics
    )

    return (
        supplier_data,
        category_metrics,
        supplier_findings,
        category_findings,
    )


def display_executive_overview(
    supplier_data: pd.DataFrame,
    category_metrics: pd.DataFrame,
) -> None:
    """
    Display headline supplier and category metrics.
    """
    total_spend = supplier_data[
        "annual_spend"
    ].sum()

    supplier_count = supplier_data[
        "supplier_id"
    ].nunique()

    category_count = supplier_data[
        "category"
    ].nunique()

    high_attention_count = (
        supplier_data["attention_level"]
        .eq("High")
        .sum()
    )

    first_row = st.columns(4)

    first_row[0].metric(
        "Total Spend",
        f"${total_spend:,.0f}",
    )

    first_row[1].metric(
        "Suppliers",
        f"{supplier_count:,}",
    )

    first_row[2].metric(
        "Categories",
        f"{category_count:,}",
    )

    first_row[3].metric(
        "High-Attention Suppliers",
        f"{high_attention_count:,}",
    )

    concentrated_categories = (
        category_metrics[
            "concentration_risk_flag"
        ]
        .sum()
    )

    fragmented_categories = (
        category_metrics[
            "fragmentation_review_flag"
        ]
        .sum()
    )

    tail_review_categories = (
        category_metrics[
            "tail_spend_review_flag"
        ]
        .sum()
    )

    second_row = st.columns(3)

    second_row[0].metric(
        "Concentration Review",
        f"{concentrated_categories:,}",
    )

    second_row[1].metric(
        "Fragmentation Review",
        f"{fragmented_categories:,}",
    )

    second_row[2].metric(
        "Tail-Spend Review",
        f"{tail_review_categories:,}",
    )


def display_supplier_attention_table(
    supplier_data: pd.DataFrame,
) -> None:
    """
    Display the top-ranked supplier attention list.
    """
    attention_columns = [
        "supplier_name",
        "category",
        "annual_spend",
        "category_spend_share_pct",
        "on_time_delivery_pct",
        "otd_change_pct_points",
        "defect_rate_pct",
        "defect_rate_change_pct_points",
        "supplier_attention_score",
        "attention_level",
        "data_confidence_pct",
    ]

    attention_table = (
        supplier_data[attention_columns]
        .sort_values(
            by="supplier_attention_score",
            ascending=False,
        )
        .head(15)
        .copy()
    )

    attention_table = attention_table.rename(
        columns={
            "supplier_name": "Supplier",
            "category": "Category",
            "annual_spend": "Annual Spend",
            "category_spend_share_pct": (
                "Category Spend Share %"
            ),
            "on_time_delivery_pct": "OTD %",
            "otd_change_pct_points": (
                "OTD Change"
            ),
            "defect_rate_pct": "Defect Rate %",
            "defect_rate_change_pct_points": (
                "Defect Rate Change"
            ),
            "supplier_attention_score": (
                "Attention Score"
            ),
            "attention_level": (
                "Attention Level"
            ),
            "data_confidence_pct": (
                "Data Confidence %"
            ),
        }
    )

    st.dataframe(
        attention_table,
        use_container_width=True,
        hide_index=True,
    )


def display_supplier_findings(
    supplier_findings: pd.DataFrame,
) -> None:
    """
    Display the top supplier management findings.
    """
    for _, finding in (
        supplier_findings.head(5).iterrows()
    ):
        with st.expander(
            (
                f"{finding['supplier_name']} — "
                f"{finding['attention_level']} attention"
            )
        ):
            st.write(
                f"**Observation:** "
                f"{finding['observation']}"
            )

            st.write(
                f"**Implication:** "
                f"{finding['implication']}"
            )

            st.write(
                f"**Suggested next step:** "
                f"{finding['suggested_next_step']}"
            )

            st.caption(
                (
                    f"Attention score: "
                    f"{finding['supplier_attention_score']} | "
                    f"Data confidence: "
                    f"{finding['data_confidence_pct']}%"
                )
            )


def display_category_findings(
    category_findings: pd.DataFrame,
) -> None:
    """
    Display category-level findings.
    """
    if category_findings.empty:
        st.info(
            "No category review findings were triggered."
        )
        return

    for _, finding in (
        category_findings.iterrows()
    ):
        with st.expander(
            str(finding["category"])
        ):
            st.write(
                f"**Observation:** "
                f"{finding['observation']}"
            )

            st.write(
                f"**Implication:** "
                f"{finding['implication']}"
            )

            st.write(
                f"**Suggested next step:** "
                f"{finding['suggested_next_step']}"
            )


def main() -> None:
    """
    Run the local Streamlit application.
    """
    st.set_page_config(
        page_title=(
            "Supplier Performance Briefing"
        ),
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "Supplier Performance & Concentration "
        "Briefing Tool"
    )

    st.write(
        "Identify supplier relationships and "
        "categories that may require management review."
    )

    st.info(
        "This prototype uses synthetic data and an "
        "illustrative scoring methodology. Findings "
        "are review hypotheses, not final sourcing decisions."
    )

    st.sidebar.header("Data source")

    use_sample_data = st.sidebar.checkbox(
        "Use sample synthetic dataset",
        value=True,
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload supplier data",
        type=["csv", "xlsx"],
        disabled=use_sample_data,
    )

    try:
        if use_sample_data:
            raw_data = load_supplier_data(
                SAMPLE_FILE
            )
        elif uploaded_file is not None:
            if uploaded_file.name.lower().endswith(
                ".csv"
            ):
                raw_data = pd.read_csv(
                    uploaded_file
                )
            else:
                raw_data = pd.read_excel(
                    uploaded_file
                )
        else:
            st.warning(
                "Upload a CSV or Excel file to continue."
            )
            st.stop()

        (
            supplier_data,
            category_metrics,
            supplier_findings,
            category_findings,
        ) = prepare_supplier_data(raw_data)

    except (
        FileNotFoundError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        st.error(
            f"Unable to process the dataset: {error}"
        )
        st.stop()

    data_quality_summary = (
        create_data_quality_summary(
            supplier_data
        )
    )

    overview_tab, supplier_tab, findings_tab = (
        st.tabs(
            [
                "Executive Overview",
                "Supplier Attention",
                "Management Findings",
            ]
        )
    )

    with overview_tab:
        st.subheader("Executive overview")

        display_executive_overview(
            supplier_data,
            category_metrics,
        )

        st.subheader("Data quality")

        quality_columns = st.columns(4)

        quality_columns[0].metric(
            "Missing Cells",
            data_quality_summary[
                "missing_cells"
            ],
        )

        quality_columns[1].metric(
            "Missing Cell %",
            (
                f"{data_quality_summary['missing_cell_pct']}%"
            ),
        )

        quality_columns[2].metric(
            "Duplicate Rows",
            data_quality_summary[
                "exact_duplicate_rows"
            ],
        )

        quality_columns[3].metric(
            "Invalid Spend Rows",
            data_quality_summary[
                "invalid_spend_rows"
            ],
        )

        st.subheader("Category overview")

        category_table = (
            category_metrics.sort_values(
                by="total_category_spend",
                ascending=False,
            )
        )

        st.dataframe(
            category_table,
            use_container_width=True,
            hide_index=True,
        )

    with supplier_tab:
        st.subheader(
            "Top supplier management-attention list"
        )

        st.caption(
            "Higher scores indicate that a supplier "
            "may deserve further management review."
        )

        display_supplier_attention_table(
            supplier_data
        )

    with findings_tab:
        st.subheader("Supplier findings")

        display_supplier_findings(
            supplier_findings
        )

        st.subheader("Category findings")

        display_category_findings(
            category_findings
        )


if __name__ == "__main__":
    main()