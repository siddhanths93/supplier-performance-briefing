from pathlib import Path

import pandas as pd
import streamlit as st

from src.category_metrics import calculate_category_metrics
from src.charts import (
    create_category_concentration_chart,
    create_supplier_risk_scatter,
)
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

from src.export import (
    create_category_briefing_export,
    create_excel_briefing_workbook,
    create_supplier_briefing_export,
)

from src.column_mapping import map_columns

from src.data_readiness import create_data_readiness_report

from src.classification import (
    classify_spend_data,
    summarize_classification_coverage,
)

from src.date_utils import (
    add_invoice_period_columns,
    summarize_date_coverage,
)

PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "synthetic_supplier_performance.csv"
)


def prepare_supplier_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict]:
    """
    Clean, validate, score, and summarize supplier data.
    """
    mapped_data, mapping_report = map_columns(raw_data)

    supplier_data = clean_supplier_data(mapped_data)

    supplier_data = add_invoice_period_columns(
        supplier_data
    )

    date_summary = summarize_date_coverage(
        supplier_data
    )

    supplier_data = classify_spend_data(
        supplier_data
    )

    supplier_data["category"] = supplier_data[
        "analysis_category"
    ]

    readiness_report = create_data_readiness_report(
        supplier_data,
        mapping_report=mapping_report,
    )

    if not readiness_report["minimum_required_ready"]:
        raise ValueError(
            "Uploaded file is missing required fields: "
            + ", ".join(
                readiness_report["missing_required_columns"]
            )
        )

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

    return (
        supplier_data,
        category_metrics,
        readiness_report,
        mapping_report,
        date_summary
    )


def display_executive_overview(
    supplier_data: pd.DataFrame,
    category_metrics: pd.DataFrame,
) -> None:
    """
    Display headline supplier and category metrics.
    """
    total_spend = supplier_data["annual_spend"].sum()

    supplier_count = supplier_data["supplier_id"].nunique()

    category_count = supplier_data["category"].nunique()

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
        category_metrics["concentration_risk_flag"]
        .sum()
    )

    fragmented_categories = (
        category_metrics["fragmentation_review_flag"]
        .sum()
    )

    tail_review_categories = (
        category_metrics["tail_spend_review_flag"]
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


def format_supplier_table(
        supplier_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a presentation-friendly supplier attention table.
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

    attention_table["annual_spend"] = (
        attention_table["annual_spend"]
        .map(lambda value: f"${value:,.0f}")
    )

    percentage_columns = [
        "category_spend_share_pct",
        "on_time_delivery_pct",
        "otd_change_pct_points",
        "defect_rate_pct",
        "defect_rate_change_pct_points",
        "data_confidence_pct",
    ]

    for column in percentage_columns:
        attention_table[column] = (
            attention_table[column]
            .map(
                lambda value: (
                    ""
                    if pd.isna(value)
                    else f"{value:.1f}%"
                )
            )
        )

    attention_table["supplier_attention_score"] = (
        attention_table["supplier_attention_score"]
        .map(lambda value: f"{value:.1f}")
    )

    attention_table = attention_table.rename(
        columns={
            "supplier_name": "Supplier",
            "category": "Category",
            "annual_spend": "Annual Spend",
            "category_spend_share_pct": (
                "Category Spend Share"
            ),
            "on_time_delivery_pct": "OTD",
            "otd_change_pct_points": "OTD Change",
            "defect_rate_pct": "Defect Rate",
            "defect_rate_change_pct_points": (
                "Defect Rate Change"
            ),
            "supplier_attention_score": (
                "Attention Score"
            ),
            "attention_level": "Attention Level",
            "data_confidence_pct": "Data Confidence",
        }
    )

    return attention_table

def display_supplier_attention_table(
    supplier_data: pd.DataFrame,
) -> None:
    """
    Display the top-ranked supplier attention list.
    """
    attention_table = format_supplier_table(
        supplier_data
    )

    st.dataframe(
        attention_table,
        use_container_width=True,
        hide_index=True,
    )

def display_supplier_detail(
    supplier_data: pd.DataFrame,
) -> None:
    """
    Display a detail panel for one selected supplier.
    """
    if supplier_data.empty:
        st.info(
            "No supplier is available for detail review."
        )
        return

    supplier_options = (
        supplier_data.sort_values(
            by="supplier_attention_score",
            ascending=False,
        )["supplier_name"]
        .tolist()
    )

    selected_supplier = st.selectbox(
        "Select a supplier to review",
        options=supplier_options,
    )

    supplier_row = supplier_data[
        supplier_data["supplier_name"]
        == selected_supplier
    ].iloc[0]

    st.markdown(
        f"### {supplier_row['supplier_name']}"
    )

    detail_columns = st.columns(4)

    detail_columns[0].metric(
        "Annual Spend",
        f"${supplier_row['annual_spend']:,.0f}",
    )

    detail_columns[1].metric(
        "Attention Score",
        f"{supplier_row['supplier_attention_score']:.1f}",
    )

    detail_columns[2].metric(
        "Attention Level",
        supplier_row["attention_level"],
    )

    detail_columns[3].metric(
        "Data Confidence",
        f"{supplier_row['data_confidence_pct']:.1f}%",
    )

    st.write("**Category:**", supplier_row["category"])
    st.write(
        "**Supplier criticality:**",
        supplier_row["supplier_criticality"],
    )

    st.write("### Performance movement")

    performance_columns = st.columns(4)

    performance_columns[0].metric(
        "Current OTD",
        (
            ""
            if pd.isna(
                supplier_row["on_time_delivery_pct"]
            )
            else f"{supplier_row['on_time_delivery_pct']:.1f}%"
        ),
    )

    performance_columns[1].metric(
        "OTD Change",
        (
            ""
            if pd.isna(
                supplier_row["otd_change_pct_points"]
            )
            else f"{supplier_row['otd_change_pct_points']:.1f}%"
        ),
    )

    performance_columns[2].metric(
        "Current Defect Rate",
        (
            ""
            if pd.isna(
                supplier_row["defect_rate_pct"]
            )
            else f"{supplier_row['defect_rate_pct']:.1f}%"
        ),
    )

    performance_columns[3].metric(
        "Defect Rate Change",
        (
            ""
            if pd.isna(
                supplier_row[
                    "defect_rate_change_pct_points"
                ]
            )
            else (
                f"{supplier_row['defect_rate_change_pct_points']:.1f}%"
            )
        ),
    )

    finding = create_supplier_findings(
        supplier_data[
            supplier_data["supplier_name"]
            == selected_supplier
        ],
        top_n=1,
    )

    if not finding.empty:
        finding_row = finding.iloc[0]

        st.write("### Evidence-backed finding")

        st.write(
            f"**Observation:** "
            f"{finding_row['observation']}"
        )

        st.write(
            f"**Implication:** "
            f"{finding_row['implication']}"
        )

        st.write(
            f"**Suggested next step:** "
            f"{finding_row['suggested_next_step']}"
        )


def display_supplier_findings(
    supplier_findings: pd.DataFrame,
) -> None:
    """
    Display the top supplier management findings.
    """
    if supplier_findings.empty:
        st.info(
            "No supplier findings match the selected filters."
        )
        return

    for _, finding in supplier_findings.head(5).iterrows():
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
            "No category review findings match the selected filters."
        )
        return

    for _, finding in category_findings.iterrows():
        with st.expander(str(finding["category"])):
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


def display_methodology() -> None:
    """
    Display the methodology and limitations used by the app.
    """
    st.subheader("Purpose")

    st.write(
        "This tool identifies supplier relationships and categories "
        "that may deserve management review based on spend exposure, "
        "supplier performance movement, category concentration, and "
        "data completeness."
    )

    st.subheader("Supplier attention score")

    st.write(
        "The supplier attention score is an illustrative, rules-based "
        "prioritization score. It is designed to help users decide where "
        "to investigate further, not to make final sourcing decisions."
    )

    methodology_table = pd.DataFrame(
        [
            {
                "Component": "Spend exposure",
                "Weight": "30%",
                "Meaning": (
                    "Higher category spend share means greater supplier "
                    "dependency or financial exposure."
                ),
            },
            {
                "Component": "Delivery risk",
                "Weight": "25%",
                "Meaning": (
                    "Suppliers with larger declines in on-time delivery "
                    "receive higher delivery risk scores."
                ),
            },
            {
                "Component": "Quality risk",
                "Weight": "25%",
                "Meaning": (
                    "Suppliers with larger increases in defect rate "
                    "receive higher quality risk scores."
                ),
            },
            {
                "Component": "Criticality",
                "Weight": "20%",
                "Meaning": (
                    "High-criticality suppliers receive higher scores "
                    "because performance issues may have greater "
                    "operational impact."
                ),
            },
        ]
    )

    st.dataframe(
        methodology_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Data confidence")

    st.write(
        "Data confidence measures how many scoring inputs are available "
        "for each supplier. Missing delivery, quality, spend, or criticality "
        "inputs reduce confidence."
    )

    st.info(
        "Missing data does not automatically increase supplier risk. "
        "Instead, the tool calculates the score from available evidence "
        "and separately reports data confidence."
    )

    st.subheader("Category concentration logic")

    st.write(
        "Category concentration and fragmentation are evaluated using "
        "transparent metrics such as top-supplier share, top-three supplier "
        "share, suppliers required to reach 80% of category spend, HHI, "
        "and tail-supplier share."
    )

    category_methodology_table = pd.DataFrame(
        [
            {
                "Metric": "Top supplier share",
                "Meaning": (
                    "The largest supplier's share of total category spend."
                ),
            },
            {
                "Metric": "Top 3 supplier share",
                "Meaning": (
                    "The combined share of the three largest suppliers "
                    "in the category."
                ),
            },
            {
                "Metric": "Suppliers to 80% spend",
                "Meaning": (
                    "The number of suppliers required to reach 80% of "
                    "category spend."
                ),
            },
            {
                "Metric": "HHI",
                "Meaning": (
                    "A concentration index calculated from squared supplier "
                    "spend shares. Higher values indicate more concentration."
                ),
            },
            {
                "Metric": "Tail supplier share",
                "Meaning": (
                    "The percentage of suppliers below the current low-spend "
                    "threshold."
                ),
            },
        ]
    )

    st.dataframe(
        category_methodology_table,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Important limitations")

    st.warning(
        "The findings are review hypotheses. They are not final sourcing "
        "decisions, guaranteed savings estimates, supplier termination "
        "recommendations, or industry-standard supplier risk ratings."
    )

    st.write(
        "This independent project uses synthetic sample data and an "
        "original illustrative scoring framework. It does not use employer, "
        "client, confidential, or proprietary data."
    )


def main() -> None:
    """
    Run the local Streamlit application.
    """
    st.set_page_config(
        page_title="Supplier Performance Briefing",
        page_icon="📊",
        layout="wide",
    )

    st.title(
        "Supplier Performance & Concentration "
        "Briefing Tool"
    )

    st.write(
        "Identify supplier relationships and categories "
        "that may require management review."
    )

    st.info(
        "This prototype uses synthetic data and an "
        "illustrative scoring methodology. Findings are "
        "review hypotheses, not final sourcing decisions."
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
            raw_data = load_supplier_data(SAMPLE_FILE)

        elif uploaded_file is not None:
            if uploaded_file.name.lower().endswith(".csv"):
                raw_data = pd.read_csv(uploaded_file)
            else:
                raw_data = pd.read_excel(uploaded_file)

        else:
            st.warning(
                "Upload a CSV or Excel file to continue."
            )
            st.stop()

        (
            supplier_data,
            category_metrics,
            readiness_report,
            mapping_report,
            date_summary,
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

    st.sidebar.header("Filters")

    available_categories = sorted(
        supplier_data["category"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "Category",
        options=available_categories,
        default=available_categories,
    )

    available_attention_levels = [
        "High",
        "Medium",
        "Low",
    ]

    selected_attention_levels = st.sidebar.multiselect(
        "Attention level",
        options=available_attention_levels,
        default=available_attention_levels,
    )

    filtered_supplier_data = supplier_data[
        supplier_data["category"].isin(
            selected_categories
        )
        & supplier_data["attention_level"].isin(
            selected_attention_levels
        )
    ].copy()

    filtered_category_metrics = category_metrics[
        category_metrics["category"].isin(
            selected_categories
        )
    ].copy()

    filtered_supplier_findings = create_supplier_findings(
        filtered_supplier_data,
        top_n=10,
    )

    filtered_category_findings = create_category_findings(
        filtered_category_metrics
    )

    classification_summary = summarize_classification_coverage(
        filtered_supplier_data
    )

    readiness_tab, overview_tab, supplier_tab, findings_tab, methodology_tab = st.tabs(
        [
            "Data Readiness",
            "Executive Overview",
            "Supplier Attention",
            "Management Findings",
            "Methodology",
        ]
    )

    with readiness_tab:
        st.subheader("Data readiness review")

        status = readiness_report["analysis_status"]

        if status == "Ready":
            st.success("Status: Ready")
        elif status == "Ready with Limitations":
            st.warning("Status: Ready with limitations")
        else:
            st.error("Status: Not ready")

        st.caption(
            "This review explains what the uploaded file can support "
            "before analytics, scoring, and findings are produced."
        )

        readiness_columns = st.columns(5)

        readiness_columns[0].metric(
            "Rows Uploaded",
            readiness_report["row_count"],
        )

        readiness_columns[1].metric(
            "Columns Detected",
            readiness_report["column_count"],
        )

        readiness_columns[2].metric(
            "File Type",
            readiness_report["input_file_type"],
        )

        readiness_columns[3].metric(
            "Mapped Columns",
            readiness_report["mapped_column_count"],
        )

        readiness_columns[4].metric(
            "Unmapped Columns",
            readiness_report["unmapped_column_count"],
        )
        st.subheader("Date coverage review")

        if date_summary["has_date_column"]:
            st.info(
                "A date column was detected. The app created invoice_year, "
                "invoice_quarter, and invoice_month fields for future "
                "time-based analysis."
            )
        else:
            st.warning(
                "No date column was detected. The app can still analyze total "
                "uploaded spend, but time-based analysis such as monthly trends "
                "or year-over-year movement will be limited."
            )

        date_columns = st.columns(5)

        date_columns[0].metric(
            "Valid Date Rows",
            date_summary["valid_date_rows"],
        )

        date_columns[1].metric(
            "Invalid/Missing Date Rows",
            date_summary["invalid_date_rows"],
        )

        date_columns[2].metric(
            "Date Coverage",
            f"{date_summary['date_coverage_pct']}%",
        )

        date_columns[3].metric(
            "Earliest Date",
            date_summary["min_date"] or "N/A",
        )

        date_columns[4].metric(
            "Latest Date",
            date_summary["max_date"] or "N/A",
        )

        classification_columns = st.columns(4)

        classification_columns[0].metric(
            "Classified Rows",
            classification_summary["classified_rows"],
        )

        classification_columns[1].metric(
            "Unclassified Rows",
            classification_summary["unclassified_rows"],
        )

        classification_columns[2].metric(
            "Review Required",
            classification_summary["review_required_rows"],
        )

        classification_columns[3].metric(
            "Classification Coverage",
            f"{classification_summary['classification_coverage_pct']}%",
        )

        st.subheader("Supported upload types")

        st.write(
            "**Supplier-level performance file:** each row represents a "
            "supplier or supplier-category relationship. This is best for "
            "full supplier attention scoring."
        )

        st.write(
            "**Transaction-level spend file:** each row represents an invoice, "
            "PO line, or spend transaction. The app can aggregate rows before "
            "analysis, but performance scoring depends on whether performance "
            "fields are included."
        )

        st.subheader("Minimum required fields")

        st.write(
            "For basic analysis, the file must include supplier/vendor name, "
            "spend amount, and either category or description."
        )

        st.subheader("Analysis capabilities")

        st.dataframe(
            readiness_report["analysis_capabilities"],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Classification sample")

        classification_columns_to_show = [
            "supplier_name",
            "description",
            "annual_spend",
            "invoice_date",
            "invoice_year",
            "invoice_quarter",
            "invoice_month",
            "taxonomy_level_1",
            "taxonomy_level_2",
            "classification_confidence",
            "classification_score",
            "classification_reason",
            "needs_classification_review",
        ]

        available_classification_columns = [
            column
            for column in classification_columns_to_show
            if column in filtered_supplier_data.columns
        ]

        st.dataframe(
            filtered_supplier_data[
                available_classification_columns
            ].head(25),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Column readiness")

        st.dataframe(
            readiness_report["column_readiness_table"],
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Column mapping report")

        st.dataframe(
            mapping_report,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("Analysis limitations")

        for limitation in readiness_report["analysis_limitations"]:
            st.write(f"- {limitation}")

    with overview_tab:
        st.subheader("Executive overview")

        if filtered_supplier_data.empty:
            st.warning(
                "No suppliers match the selected filters."
            )
        else:
            display_executive_overview(
                filtered_supplier_data,
                filtered_category_metrics,
            )

            data_quality_summary = (
                create_data_quality_summary(
                    filtered_supplier_data
                )
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

        if filtered_category_metrics.empty:
            st.warning(
                "No categories match the selected filters."
            )
        else:
            category_table = (
                filtered_category_metrics.sort_values(
                    by="total_category_spend",
                    ascending=False,
                )
            )

            st.dataframe(
                category_table,
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("Category concentration")

            concentration_chart = (
                create_category_concentration_chart(
                    filtered_category_metrics
                )
            )

            st.plotly_chart(
                concentration_chart,
                use_container_width=True,
            )

    with supplier_tab:
        st.subheader(
            "Top supplier management-attention list"
        )

        st.caption(
            "Higher scores indicate that a supplier may "
            "deserve further management review."
        )

        if filtered_supplier_data.empty:
            st.warning(
                "No suppliers match the selected filters."
            )
        else:
            st.caption(
                f"Showing {len(filtered_supplier_data)} "
                f"filtered suppliers."
            )

            st.subheader("Supplier detail")

            display_supplier_detail(
                filtered_supplier_data
            )

            st.subheader(
                "Supplier spend and attention positioning"
            )


            supplier_scatter = create_supplier_risk_scatter(
                filtered_supplier_data
            )

            st.plotly_chart(
                supplier_scatter,
                use_container_width=True,
            )

    with findings_tab:
        st.subheader("Supplier findings")

        display_supplier_findings(
            filtered_supplier_findings
        )

        supplier_export = create_supplier_briefing_export(
            filtered_supplier_findings
        )

        if not supplier_export.empty:
            st.download_button(
                label="Download supplier briefing CSV",
                data=supplier_export.to_csv(index=False),
                file_name="supplier_briefing.csv",
                mime="text/csv",
            )

        st.subheader("Category findings")

        display_category_findings(
            filtered_category_findings
        )

        category_export = create_category_briefing_export(
            filtered_category_findings
        )

        if category_export.empty:
            st.info(
                "No category findings are available for the selected filters, "
                "so there is no category briefing to download."
            )
        else:
            st.download_button(
                label="Download category briefing CSV",
                data=category_export.to_csv(index=False),
                file_name="category_briefing.csv",
                mime="text/csv",
            )

        st.subheader("Executive briefing workbook")

        excel_briefing = create_excel_briefing_workbook(
            supplier_findings=filtered_supplier_findings,
            category_findings=filtered_category_findings,
            supplier_data=filtered_supplier_data,
            category_metrics=filtered_category_metrics,
        )

        st.download_button(
            label="Download executive briefing Excel workbook",
            data=excel_briefing,
            file_name="supplier_executive_briefing.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

    with methodology_tab:
        display_methodology()

if __name__ == "__main__":
    main()