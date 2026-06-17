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

from src.aggregation import (
    aggregate_supplier_category_data,
    create_aggregation_summary,
    should_aggregate_supplier_data,
)

from src.schema import ensure_optional_analysis_columns


def apply_custom_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1400px;
        }

        h1 {
            font-size: 2.4rem !important;
            font-weight: 800 !important;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        h2, h3 {
            color: #0f172a;
            font-weight: 750 !important;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 1rem 1rem;
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            color: #64748b;
            font-weight: 650;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            color: #0f172a;
            font-weight: 800;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0e7490 100%);
            color: white;
            border-radius: 22px;
            padding: 1.6rem 1.8rem;
            margin-bottom: 1.5rem;
        }

        .hero-card h2 {
            color: white;
            margin-bottom: 0.4rem;
        }

        .hero-card p {
            color: #dbeafe;
            font-size: 1rem;
            line-height: 1.6;
        }

        .status-pill-ready {
            display: inline-block;
            background: #dcfce7;
            color: #166534;
            border: 1px solid #86efac;
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .status-pill-warning {
            display: inline-block;
            background: #fef3c7;
            color: #92400e;
            border: 1px solid #fcd34d;
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .status-pill-error {
            display: inline-block;
            background: #fee2e2;
            color: #991b1b;
            border: 1px solid #fca5a5;
            border-radius: 999px;
            padding: 0.45rem 0.85rem;
            font-weight: 700;
            font-size: 0.9rem;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .insight-box {
            background: #eff6ff;
            border-left: 5px solid #2563eb;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            color: #1e3a8a;
            margin: 0.75rem 0 1rem 0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.35rem;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 0.5rem 1rem;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
        }

        .stTabs [aria-selected="true"] {
            background: #0f172a !important;
            color: white !important;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
            border: 1px solid #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value):
    try:
        return f"${float(value):,.0f}"
    except Exception:
        return "N/A"


def format_number(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "N/A"


def format_percent(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "N/A"


def prettify_column_name(column_name: str) -> str:
    return (
        column_name.replace("_", " ")
        .replace("pct", "%")
        .title()
        .replace("Otd", "OTD")
        .replace("Po ", "PO ")
        .replace("Gl ", "GL ")
    )


def make_display_table(data):
    display_data = data.copy()
    display_data.columns = [
        prettify_column_name(column)
        for column in display_data.columns
    ]
    return display_data



PROJECT_ROOT = Path(__file__).resolve().parent

SAMPLE_FILE = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "synthetic_supplier_performance.csv"
)


def prepare_supplier_data(
    raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict, dict]:
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

    aggregation_time_grain = None

    pre_aggregation_data = supplier_data.copy()

    was_aggregated = should_aggregate_supplier_data(
        supplier_data,
        time_grain=aggregation_time_grain,
    )

    if was_aggregated:
        supplier_data = aggregate_supplier_category_data(
            supplier_data,
            time_grain=aggregation_time_grain,
        )

    aggregation_summary = create_aggregation_summary(
        before_data=pre_aggregation_data,
        after_data=supplier_data,
        was_aggregated=was_aggregated,
        time_grain=aggregation_time_grain,
    )

    supplier_data = ensure_optional_analysis_columns(
        supplier_data
    )

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
        date_summary,
        aggregation_summary,
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
        page_title="Supplier Performance & Spend Intelligence",
        layout="wide",
    )

    apply_custom_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h2>Supplier Performance & Spend Intelligence Dashboard</h2>
            <p>
            Upload supplier spend or performance data to assess data readiness, classify spend,
            identify concentration risks, score supplier attention areas, and generate executive-ready outputs.
            </p>
            <p>
            <strong>Portfolio demo:</strong> built with Python, Streamlit, Pandas, Plotly, OpenPyXL, and Pytest.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Upload Data")

    st.sidebar.write(
        "Upload a supplier-level performance file or transaction-level spend file. "
        "The app maps common column names automatically and explains which analyses are available."
    )

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
            aggregation_summary
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
            "1. Data Readiness",
            "2. Executive Overview",
            "3. Supplier Attention",
            "4. Insights Briefing",
            "5. Methodology",
        ]
    )

    with readiness_tab:
        st.subheader("Data readiness review")

        status = readiness_report["analysis_status"]

        if status == "Ready":
            status_class = "status-pill-ready"
        elif status == "Ready with Limitations":
            status_class = "status-pill-warning"
        else:
            status_class = "status-pill-error"

        st.markdown(
            f"""
            <div class="section-card">
                <span class="{status_class}">Status: {status}</span>
                <p class="small-muted" style="margin-top: 0.85rem;">
                This tab checks whether the uploaded file is ready for analysis. It shows which columns were mapped,
                which fields are missing, whether dates were usable, how spend was classified, and whether transaction
                rows were aggregated before scoring.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        readiness_columns = st.columns(5)

        readiness_columns[0].metric(
            "Rows analyzed",
            format_number(readiness_report["row_count"]),
        )

        readiness_columns[1].metric(
            "Columns detected",
            format_number(readiness_report["column_count"]),
        )

        readiness_columns[2].metric(
            "Mapped columns",
            format_number(readiness_report["mapped_column_count"]),
        )

        readiness_columns[3].metric(
            "Unmapped columns",
            format_number(readiness_report["unmapped_column_count"]),
        )

        readiness_columns[4].metric(
            "File type",
            readiness_report["input_file_type"].replace(" file", ""),
        )

        st.divider()

        st.subheader("Date coverage")

        if date_summary["has_date_column"]:
            st.markdown(
                """
                <div class="insight-box">
                A date column was detected. The app created year, quarter, and month fields for future time-based analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "No date column was detected. The app can still analyze total uploaded spend, "
                "but time-based analysis will be limited."
            )

        date_columns = st.columns(5)

        date_columns[0].metric(
            "Valid date rows",
            format_number(date_summary["valid_date_rows"]),
        )

        date_columns[1].metric(
            "Invalid/missing dates",
            format_number(date_summary["invalid_date_rows"]),
        )

        date_columns[2].metric(
            "Date coverage",
            format_percent(date_summary["date_coverage_pct"]),
        )

        date_columns[3].metric(
            "Earliest date",
            date_summary["min_date"] or "N/A",
        )

        date_columns[4].metric(
            "Latest date",
            date_summary["max_date"] or "N/A",
        )

        st.divider()

        st.subheader("Classification coverage")

        classification_columns = st.columns(4)

        classification_columns[0].metric(
            "Classified rows",
            format_number(classification_summary["classified_rows"]),
        )

        classification_columns[1].metric(
            "Unclassified rows",
            format_number(classification_summary["unclassified_rows"]),
        )

        classification_columns[2].metric(
            "Review required",
            format_number(classification_summary["review_required_rows"]),
        )

        classification_columns[3].metric(
            "Coverage",
            format_percent(classification_summary["classification_coverage_pct"]),
        )

        st.divider()

        st.subheader("Aggregation review")

        if aggregation_summary["was_aggregated"]:
            st.markdown(
                """
                <div class="insight-box">
                Multiple rows for the same supplier/category were detected, so the app aggregated them before scoring and dashboard analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.success(
                "No supplier/category aggregation was required for the main dashboard view."
            )

        aggregation_columns = st.columns(5)

        aggregation_columns[0].metric(
            "Aggregation grain",
            aggregation_summary["time_grain"],
        )

        aggregation_columns[1].metric(
            "Input rows",
            format_number(aggregation_summary["input_rows"]),
        )

        aggregation_columns[2].metric(
            "Rows after aggregation",
            format_number(aggregation_summary["output_rows"]),
        )

        aggregation_columns[3].metric(
            "Rows reduced",
            format_number(aggregation_summary["rows_reduced"]),
        )

        aggregation_columns[4].metric(
            "Suppliers detected",
            format_number(aggregation_summary["output_supplier_count"]),
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

        st.divider()

        st.subheader("Classification sample")

        classification_columns_to_show = [
            "supplier_name",
            "description",
            "annual_spend",
            "invoice_date",
            "taxonomy_level_1",
            "taxonomy_level_2",
            "classification_confidence",
            "classification_score",
            "needs_classification_review",
        ]

        available_classification_columns = [
            column
            for column in classification_columns_to_show
            if column in filtered_supplier_data.columns
        ]

        classification_sample = filtered_supplier_data[
            available_classification_columns
        ].head(12)

        st.dataframe(
            make_display_table(classification_sample),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("View analysis capability details"):
            st.dataframe(
                make_display_table(readiness_report["analysis_capabilities"]),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("View column readiness details"):
            st.dataframe(
                make_display_table(readiness_report["column_readiness_table"]),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("View column mapping report"):
            st.dataframe(
                make_display_table(mapping_report),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("View analysis limitations"):
            for limitation in readiness_report["analysis_limitations"]:
                st.write(f"- {limitation}")

    with overview_tab:
        st.subheader("Executive overview")

        total_spend = (
            filtered_supplier_data["annual_spend"].sum()
            if "annual_spend" in filtered_supplier_data.columns
            else 0
        )

        supplier_count = (
            filtered_supplier_data["supplier_name"].nunique()
            if "supplier_name" in filtered_supplier_data.columns
            else 0
        )

        category_count = (
            filtered_supplier_data["category"].nunique()
            if "category" in filtered_supplier_data.columns
            else 0
        )

        high_attention_count = (
            (filtered_supplier_data["attention_level"] == "High").sum()
            if "attention_level" in filtered_supplier_data.columns
            else 0
        )

        overview_columns = st.columns(4)

        overview_columns[0].metric(
            "Total spend",
            format_currency(total_spend),
        )

        overview_columns[1].metric(
            "Suppliers",
            format_number(supplier_count),
        )

        overview_columns[2].metric(
            "Categories",
            format_number(category_count),
        )

        overview_columns[3].metric(
            "High-attention suppliers",
            format_number(high_attention_count),
        )

        st.markdown(
            """
            <div class="insight-box">
            This overview summarizes supplier spend, concentration exposure, data quality, 
            and high-attention suppliers after mapping, classification, date handling, and aggregation.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        st.subheader("Data quality snapshot")

        missing_cells = (
            filtered_supplier_data.isna().sum().sum()
        )

        total_cells = (
                filtered_supplier_data.shape[0]
                * filtered_supplier_data.shape[1]
        )

        missing_cell_pct = (
            missing_cells / total_cells * 100
            if total_cells > 0
            else 0
        )

        duplicate_rows = (
            filtered_supplier_data.duplicated().sum()
        )

        invalid_spend_rows = (
            (filtered_supplier_data["annual_spend"].isna()).sum()
            if "annual_spend" in filtered_supplier_data.columns
            else 0
        )

        quality_columns = st.columns(4)

        quality_columns[0].metric(
            "Missing cells",
            format_number(missing_cells),
        )

        quality_columns[1].metric(
            "Missing cell %",
            format_percent(missing_cell_pct),
        )

        quality_columns[2].metric(
            "Duplicate rows",
            format_number(duplicate_rows),
        )

        quality_columns[3].metric(
            "Invalid spend rows",
            format_number(invalid_spend_rows),
        )

        st.divider()

        st.subheader("Category overview")

        category_display_columns = [
            "category",
            "total_category_spend",
            "supplier_count",
            "top_supplier_share_pct",
            "tail_supplier_count",
            "concentration_risk_flag",
            "fragmentation_review_flag",
            "tail_spend_review_flag",
        ]

        available_category_display_columns = [
            column
            for column in category_display_columns
            if column in filtered_category_metrics.columns
        ]

        if available_category_display_columns:
            category_overview_display = filtered_category_metrics[
                available_category_display_columns
            ].copy()

            # Keep the default view clean for interview/demo.
            st.dataframe(
                make_display_table(
                    category_overview_display.head(10)
                ),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("View full category metrics"):
                st.dataframe(
                    make_display_table(filtered_category_metrics),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning(
                "Category metrics are not available for the current uploaded file."
            )

        st.divider()

        st.subheader("Supplier attention snapshot")

        supplier_display_columns = [
            "supplier_name",
            "category",
            "annual_spend",
            "attention_score",
            "attention_level",
            "spend_change_pct",
            "on_time_delivery_pct",
            "defect_rate_pct",
            "supplier_criticality",
        ]

        available_supplier_display_columns = [
            column
            for column in supplier_display_columns
            if column in filtered_supplier_data.columns
        ]

        if available_supplier_display_columns:
            supplier_attention_display = filtered_supplier_data[
                available_supplier_display_columns
            ].copy()

            if "attention_score" in supplier_attention_display.columns:
                supplier_attention_display = supplier_attention_display.sort_values(
                    by="attention_score",
                    ascending=False,
                )

            st.dataframe(
                make_display_table(
                    supplier_attention_display.head(10)
                ),
                use_container_width=True,
                hide_index=True,
            )

            with st.expander("View full supplier dataset"):
                st.dataframe(
                    make_display_table(filtered_supplier_data),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.warning(
                "Supplier attention details are not available for the current uploaded file."
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

        st.subheader("What this project demonstrates")

        st.write(
            """
            This project is designed as a realistic supplier analytics workflow rather than a static dashboard.
            It handles messy uploaded files, maps common procurement column names, checks readiness before analysis,
            classifies spend using transparent rules, parses transaction dates, aggregates supplier/category rows,
            and produces management-style findings.
            """
        )

        st.subheader("Core capabilities")

        st.markdown(
            """
            - **Flexible file ingestion:** accepts CSV and Excel files with varied column names.
            - **Column mapping:** maps aliases like Vendor Name, Invoice Amount, OTD %, PO Number, and Cost Center to canonical fields.
            - **Data readiness:** explains which analyses are available, limited, or unavailable.
            - **Spend classification:** uses a built-in taxonomy and scored rules-based classification.
            - **Date handling:** parses invoice dates and creates year, quarter, and month fields.
            - **Supplier/category aggregation:** supports transaction-level and supplier-level files.
            - **Attention scoring:** highlights suppliers needing management review.
            - **Executive exports:** generates downloadable CSV and Excel outputs.
            """
        )

        st.subheader("Why I built it")

        st.write(
            """
            I built this to combine my procurement and supply chain analytics background with hands-on Python product development.
            The goal was to create a practical tool that mirrors the messy reality of supplier data: inconsistent column names,
            missing optional fields, extra ERP columns, mixed transaction and supplier-level files, and varying data quality.
            """
        )

if __name__ == "__main__":
    main()