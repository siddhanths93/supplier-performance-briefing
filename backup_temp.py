from __future__ import annotations

from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st

from src.aggregation import (
    aggregate_supplier_category_data,
    create_aggregation_summary,
    should_aggregate_supplier_data,
)
from src.classification import (
    classify_spend_data,
    summarize_classification_coverage,
)
from src.cleaning import clean_supplier_data
from src.column_mapping import map_columns
from src.data_readiness import create_data_readiness_report
from src.date_utils import (
    add_invoice_period_columns,
    summarize_date_coverage,
)
from src.schema import ensure_optional_analysis_columns

st.set_page_config(
    page_title="Supplier Performance & Spend Intelligence",
    page_icon="📊",
    layout="wide",
)


# -------------------------------------------------------------------
# Styling
# -------------------------------------------------------------------

def apply_custom_styles():
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1450px;
        }

        h1, h2, h3 {
            color: #0f172a;
            font-weight: 800 !important;
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            padding: 1rem 1rem;
            border-radius: 16px;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
        }

        div[data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            color: #64748b;
            font-weight: 700;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.45rem;
            color: #0f172a;
            font-weight: 850;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0e7490 100%);
            color: white;
            border-radius: 24px;
            padding: 1.8rem 2rem;
            margin-bottom: 1.4rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.20);
        }

        .hero-card h2 {
            color: white;
            margin-bottom: 0.5rem;
            font-size: 2rem;
        }

        .hero-card p {
            color: #dbeafe;
            font-size: 1rem;
            line-height: 1.6;
        }

        .section-card {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 18px;
            padding: 1.2rem 1.35rem;
            box-shadow: 0 1px 4px rgba(15, 23, 42, 0.07);
            margin-bottom: 1rem;
        }

        .section-card h4 {
            color: #0f172a;
            margin-bottom: 0.55rem;
            font-size: 1.05rem;
        }

        .section-card p {
            color: #334155;
            line-height: 1.55;
            margin-bottom: 0.45rem;
        }

        .insight-box {
            background: #eff6ff;
            border-left: 5px solid #2563eb;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            color: #1e3a8a;
            margin: 0.75rem 0 1rem 0;
        }

        .warning-box {
            background: #fffbeb;
            border-left: 5px solid #f59e0b;
            padding: 1rem 1.2rem;
            border-radius: 12px;
            color: #92400e;
            margin: 0.75rem 0 1rem 0;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
            line-height: 1.55;
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


# -------------------------------------------------------------------
# Formatting helpers
# -------------------------------------------------------------------

def format_currency(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"${float(value):,.0f}"
    except Exception:
        return "N/A"


def format_number(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.0f}"
    except Exception:
        return "N/A"


def format_percent(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.1f}%"
    except Exception:
        return "N/A"


def prettify_column_name(column_name: str) -> str:
    return (
        str(column_name)
        .replace("_", " ")
        .replace("pct", "%")
        .title()
        .replace("Otd", "OTD")
        .replace("Po ", "PO ")
        .replace("Gl ", "GL ")
        .replace("Id", "ID")
    )


def make_display_table(data: pd.DataFrame) -> pd.DataFrame:
    display_data = data.copy()
    display_data.columns = [
        prettify_column_name(column)
        for column in display_data.columns
    ]
    return display_data


# -------------------------------------------------------------------
# Chart helpers
# -------------------------------------------------------------------

def apply_chart_layout(fig, height: int = 420):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=60, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        title=dict(font=dict(size=18)),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
    )
    return fig


def show_chart_or_message(fig, message: str):
    if fig is None:
        st.info(message)
    else:
        st.plotly_chart(fig, use_container_width=True)


def create_top_suppliers_chart(data: pd.DataFrame):
    if (
            data.empty
            or "supplier_name" not in data.columns
            or "annual_spend" not in data.columns
    ):
        return None

    chart_data = (
        data.groupby("supplier_name", dropna=False)["annual_spend"]
        .sum()
        .reset_index()
        .sort_values(by="annual_spend", ascending=False)
        .head(10)
        .sort_values(by="annual_spend", ascending=True)
    )

    fig = px.bar(
        chart_data,
        x="annual_spend",
        y="supplier_name",
        orientation="h",
        title="Top 10 Suppliers by Spend",
        labels={
            "annual_spend": "Spend",
            "supplier_name": "Supplier",
        },
    )

    return apply_chart_layout(fig)


def create_top_categories_chart(data: pd.DataFrame):
    if (
            data.empty
            or "category" not in data.columns
            or "annual_spend" not in data.columns
    ):
        return None

    chart_data = (
        data.groupby("category", dropna=False)["annual_spend"]
        .sum()
        .reset_index()
        .sort_values(by="annual_spend", ascending=False)
        .head(10)
        .sort_values(by="annual_spend", ascending=True)
    )

    fig = px.bar(
        chart_data,
        x="annual_spend",
        y="category",
        orientation="h",
        title="Top Categories by Spend",
        labels={
            "annual_spend": "Spend",
            "category": "Category",
        },
    )

    return apply_chart_layout(fig)


def create_spend_by_region_chart(data: pd.DataFrame):
    if (
            data.empty
            or "region" not in data.columns
            or "annual_spend" not in data.columns
    ):
        return None

    chart_data = (
        data.dropna(subset=["region"])
        .groupby("region", dropna=False)["annual_spend"]
        .sum()
        .reset_index()
        .sort_values(by="annual_spend", ascending=False)
    )

    if chart_data.empty:
        return None

    fig = px.bar(
        chart_data,
        x="region",
        y="annual_spend",
        title="Spend by Region",
        labels={
            "region": "Region",
            "annual_spend": "Spend",
        },
    )

    return apply_chart_layout(fig)


def create_spend_by_business_unit_chart(data: pd.DataFrame):
    if (
            data.empty
            or "business_unit" not in data.columns
            or "annual_spend" not in data.columns
    ):
        return None

    chart_data = (
        data.dropna(subset=["business_unit"])
        .groupby("business_unit", dropna=False)["annual_spend"]
        .sum()
        .reset_index()
        .sort_values(by="annual_spend", ascending=False)
    )

    if chart_data.empty:
        return None

    fig = px.bar(
        chart_data,
        x="business_unit",
        y="annual_spend",
        title="Spend by Business Unit",
        labels={
            "business_unit": "Business Unit",
            "annual_spend": "Spend",
        },
    )

    return apply_chart_layout(fig)


def create_monthly_spend_trend_chart(data: pd.DataFrame):
    if (
            data.empty
            or "invoice_month" not in data.columns
            or "annual_spend" not in data.columns
    ):
        return None

    chart_data = data.dropna(subset=["invoice_month"]).copy()

    if chart_data.empty:
        return None

    chart_data = (
        chart_data.groupby("invoice_month", dropna=False)["annual_spend"]
        .sum()
        .reset_index()
        .sort_values(by="invoice_month")
    )

    fig = px.line(
        chart_data,
        x="invoice_month",
        y="annual_spend",
        markers=True,
        title="Spend Trend Over Time",
        labels={
            "invoice_month": "Month",
            "annual_spend": "Spend",
        },
    )

    return apply_chart_layout(fig)


def create_supplier_count_by_category_chart(data: pd.DataFrame):
    if (
            data.empty
            or "category" not in data.columns
            or "supplier_name" not in data.columns
    ):
        return None

    chart_data = (
        data.groupby("category", dropna=False)["supplier_name"]
        .nunique()
        .reset_index(name="supplier_count")
        .sort_values(by="supplier_count", ascending=False)
        .head(10)
        .sort_values(by="supplier_count", ascending=True)
    )

    fig = px.bar(
        chart_data,
        x="supplier_count",
        y="category",
        orientation="h",
        title="Supplier Count by Category",
        labels={
            "supplier_count": "Supplier Count",
            "category": "Category",
        },
    )

    return apply_chart_layout(fig)


def create_opportunity_savings_chart(opportunities: pd.DataFrame):
    if opportunities.empty:
        return None

    required_columns = {
        "category",
        "estimated_savings_low",
        "estimated_savings_high",
    }

    if not required_columns.issubset(opportunities.columns):
        return None

    chart_data = (
        opportunities.groupby("category", dropna=False)[
            ["estimated_savings_low", "estimated_savings_high"]
        ]
        .sum()
        .reset_index()
        .sort_values(by="estimated_savings_high", ascending=False)
        .head(10)
        .sort_values(by="estimated_savings_high", ascending=True)
    )

    fig = px.bar(
        chart_data,
        x=["estimated_savings_low", "estimated_savings_high"],
        y="category",
        orientation="h",
        barmode="group",
        title="Estimated Savings Range by Category",
        labels={
            "value": "Estimated Savings",
            "category": "Category",
            "variable": "Savings Estimate",
        },
    )

    return apply_chart_layout(fig)


def create_opportunity_type_chart(opportunities: pd.DataFrame):
    if opportunities.empty or "opportunity_type" not in opportunities.columns:
        return None

    chart_data = (
        opportunities.groupby("opportunity_type", dropna=False)
        .size()
        .reset_index(name="opportunity_count")
        .sort_values(by="opportunity_count", ascending=False)
    )

    fig = px.pie(
        chart_data,
        names="opportunity_type",
        values="opportunity_count",
        title="Opportunity Mix",
    )

    return apply_chart_layout(fig)


def create_priority_chart(opportunities: pd.DataFrame):
    if opportunities.empty or "priority" not in opportunities.columns:
        return None

    chart_data = (
        opportunities.groupby("priority", dropna=False)
        .size()
        .reset_index(name="opportunity_count")
    )

    fig = px.bar(
        chart_data,
        x="priority",
        y="opportunity_count",
        title="Opportunities by Priority",
        labels={
            "priority": "Priority",
            "opportunity_count": "Opportunity Count",
        },
    )

    return apply_chart_layout(fig)


# -------------------------------------------------------------------
# Procurement opportunity engine
# -------------------------------------------------------------------

def assign_priority(
        savings_amount: float,
        supplier_count: int,
        opportunity_type: str,
) -> str:
    if savings_amount >= 10000 or supplier_count >= 5:
        return "High"

    if savings_amount >= 3000 or supplier_count >= 3:
        return "Medium"

    return "Low"


def estimate_savings_pct(
        opportunity_type: str,
        supplier_count: int,
        contract_gap_pct: float = 0,
) -> tuple[float, float]:
    if opportunity_type == "Fragmented category":
        if supplier_count >= 8:
            return 0.07, 0.12
        if supplier_count >= 5:
            return 0.05, 0.09
        return 0.03, 0.06

    if opportunity_type == "Tail spend cleanup":
        return 0.03, 0.08

    if opportunity_type == "No contract coverage":
        if contract_gap_pct >= 50:
            return 0.06, 0.10
        return 0.04, 0.07

    if opportunity_type == "Supplier consolidation":
        return 0.04, 0.10

    return 0.02, 0.05


def create_rationalization_opportunities(
        supplier_data: pd.DataFrame,
) -> pd.DataFrame:
    data = supplier_data.copy()

    if (
            data.empty
            or "category" not in data.columns
            or "annual_spend" not in data.columns
            or "supplier_name" not in data.columns
    ):
        return pd.DataFrame()

    opportunities = []

    grouped = data.groupby("category", dropna=False)

    for category, category_data in grouped:
        total_spend = float(
            pd.to_numeric(
                category_data["annual_spend"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        supplier_count = int(category_data["supplier_name"].nunique())

        if total_spend <= 0:
            continue

        low_spend_suppliers = category_data[
            pd.to_numeric(
                category_data["annual_spend"],
                errors="coerce",
            ).fillna(0)
            < 10000
            ]

        tail_spend = float(
            pd.to_numeric(
                low_spend_suppliers["annual_spend"],
                errors="coerce",
            )
            .fillna(0)
            .sum()
        )

        tail_supplier_count = int(
            low_spend_suppliers["supplier_name"].nunique()
        )

        no_contract_count = 0

        if "contract_status" in category_data.columns:
            no_contract_count = int(
                category_data["contract_status"]
                .fillna("")
                .astype(str)
                .str.lower()
                .isin(
                    [
                        "no contract",
                        "not contracted",
                        "none",
                        "expired",
                        "missing",
                    ]
                )
                .sum()
            )

        contract_gap_pct = (
            no_contract_count / len(category_data) * 100
            if len(category_data) > 0
            else 0
        )

        if supplier_count >= 4:
            low_pct, high_pct = estimate_savings_pct(
                "Fragmented category",
                supplier_count,
            )

            savings_low = total_spend * low_pct
            savings_high = total_spend * high_pct

            suggested_reduction = max(
                supplier_count - max(2, round(supplier_count * 0.6)),
                1,
            )

            opportunities.append(
                {
                    "opportunity_type": "Fragmented category",
                    "category": category,
                    "current_suppliers": supplier_count,
                    "total_spend": total_spend,
                    "suggested_supplier_reduction": suggested_reduction,
                    "rationale": (
                        f"{category} has {supplier_count} suppliers, indicating potential fragmentation."
                    ),
                    "estimated_savings_low": savings_low,
                    "estimated_savings_high": savings_high,
                    "estimated_savings_range": (
                        f"${savings_low:,.0f} - ${savings_high:,.0f}"
                    ),
                    "priority": assign_priority(
                        savings_high,
                        supplier_count,
                        "Fragmented category",
                    ),
                    "next_action": (
                        f"Review supplier overlap in {category} and identify preferred suppliers."
                    ),
                }
            )

        if tail_supplier_count >= 3 and tail_spend > 0:
            low_pct, high_pct = estimate_savings_pct(
                "Tail spend cleanup",
                tail_supplier_count,
            )

            savings_low = tail_spend * low_pct
            savings_high = tail_spend * high_pct

            opportunities.append(
                {
                    "opportunity_type": "Tail spend cleanup",
                    "category": category,
                    "current_suppliers": tail_supplier_count,
                    "total_spend": tail_spend,
                    "suggested_supplier_reduction": max(
                        round(tail_supplier_count * 0.5),
                        1,
                    ),
                    "rationale": (
                        f"{category} has {tail_supplier_count} low-spend suppliers under $10K."
                    ),
                    "estimated_savings_low": savings_low,
                    "estimated_savings_high": savings_high,
                    "estimated_savings_range": (
                        f"${savings_low:,.0f} - ${savings_high:,.0f}"
                    ),
                    "priority": assign_priority(
                        savings_high,
                        tail_supplier_count,
                        "Tail spend cleanup",
                    ),
                    "next_action": (
                        f"Consolidate low-spend suppliers in {category} into preferred suppliers or catalogs."
                    ),
                }
            )

        if no_contract_count >= 2:
            low_pct, high_pct = estimate_savings_pct(
                "No contract coverage",
                supplier_count,
                contract_gap_pct,
            )

            savings_low = total_spend * low_pct
            savings_high = total_spend * high_pct

            opportunities.append(
                {
                    "opportunity_type": "No contract coverage",
                    "category": category,
                    "current_suppliers": supplier_count,
                    "total_spend": total_spend,
                    "suggested_supplier_reduction": 0,
                    "rationale": (
                        f"{no_contract_count} supplier records in {category} appear to lack active contract coverage."
                    ),
                    "estimated_savings_low": savings_low,
                    "estimated_savings_high": savings_high,
                    "estimated_savings_range": (
                        f"${savings_low:,.0f} - ${savings_high:,.0f}"
                    ),
                    "priority": assign_priority(
                        savings_high,
                        supplier_count,
                        "No contract coverage",
                    ),
                    "next_action": (
                        f"Review contract status and negotiate pricing or terms for uncovered {category} suppliers."
                    ),
                }
            )

    opportunities_data = pd.DataFrame(opportunities)

    if opportunities_data.empty:
        return opportunities_data

    priority_order = {
        "High": 1,
        "Medium": 2,
        "Low": 3,
    }

    opportunities_data["priority_sort"] = opportunities_data["priority"].map(
        priority_order
    )

    opportunities_data = (
        opportunities_data.sort_values(
            by=["priority_sort", "estimated_savings_high"],
            ascending=[True, False],
        )
        .drop(columns=["priority_sort"])
        .reset_index(drop=True)
    )

    return opportunities_data


def summarize_opportunity_pipeline(
        opportunities: pd.DataFrame,
) -> dict:
    if opportunities.empty:
        return {
            "opportunity_count": 0,
            "high_priority_count": 0,
            "estimated_savings_low": 0,
            "estimated_savings_high": 0,
            "top_category": "N/A",
        }

    high_priority_count = int(
        (opportunities["priority"] == "High").sum()
    )

    top_category = (
        opportunities.sort_values(
            by="estimated_savings_high",
            ascending=False,
        )
        .iloc[0]["category"]
    )

    return {
        "opportunity_count": len(opportunities),
        "high_priority_count": high_priority_count,
        "estimated_savings_low": float(
            opportunities["estimated_savings_low"].sum()
        ),
        "estimated_savings_high": float(
            opportunities["estimated_savings_high"].sum()
        ),
        "top_category": top_category,
    }


def create_procurement_recommendations(
        opportunities: pd.DataFrame,
) -> list[dict]:
    if opportunities.empty:
        return [
            {
                "recommendation": (
                    "Improve spend file completeness before deeper sourcing analysis."
                ),
                "why_it_matters": (
                    "The uploaded file may have limited category, supplier, contract, or transaction detail."
                ),
                "expected_benefit": (
                    "Better classification, richer sourcing insights, and stronger savings estimates."
                ),
                "next_action": (
                    "Upload category, business unit, region, contract status, payment terms, and transaction date where available."
                ),
                "difficulty": "Low",
                "priority": "Medium",
                "estimated_savings_range": "Not estimated",
            }
        ]

    recommendations = []

    for _, row in opportunities.head(8).iterrows():
        category = row["category"]
        opportunity_type = row["opportunity_type"]

        if opportunity_type == "Fragmented category":
            recommendation = (
                f"Rationalize suppliers in {category} by narrowing the supplier base "
                f"from {row['current_suppliers']} suppliers."
            )
            benefit = (
                "Reduced fragmentation, stronger negotiation leverage, and easier supplier governance."
            )
            difficulty = "Medium"

        elif opportunity_type == "Tail spend cleanup":
            recommendation = (
                f"Clean up tail spend in {category} by consolidating low-spend suppliers."
            )
            benefit = (
                "Lower administrative effort, reduced supplier count, and improved catalog compliance."
            )
            difficulty = "Low"

        elif opportunity_type == "No contract coverage":
            recommendation = (
                f"Review contract coverage in {category} and negotiate pricing or terms where contracts are missing."
            )
            benefit = (
                "Improved commercial control and reduced unmanaged spend exposure."
            )
            difficulty = "Medium"

        else:
            recommendation = f"Review sourcing opportunity in {category}."
            benefit = "Potential savings and better supplier management."
            difficulty = "Medium"

        recommendations.append(
            {
                "recommendation": recommendation,
                "why_it_matters": row["rationale"],
                "expected_benefit": benefit,
                "next_action": row["next_action"],
                "difficulty": difficulty,
                "priority": row["priority"],
                "estimated_savings_range": row["estimated_savings_range"],
            }
        )

    return recommendations


# -------------------------------------------------------------------
# Processing helpers
# -------------------------------------------------------------------

def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")


def create_category_metrics(data: pd.DataFrame) -> pd.DataFrame:
    if (
            data.empty
            or "category" not in data.columns
            or "annual_spend" not in data.columns
            or "supplier_name" not in data.columns
    ):
        return pd.DataFrame()

    grouped = data.groupby("category", dropna=False)

    category_metrics = grouped.agg(
        total_category_spend=("annual_spend", "sum"),
        supplier_count=("supplier_name", "nunique"),
        transaction_count=("supplier_name", "count"),
    ).reset_index()

    total_spend = category_metrics["total_category_spend"].sum()

    category_metrics["category_spend_share_pct"] = (
        category_metrics["total_category_spend"] / total_spend * 100
        if total_spend > 0
        else 0
    )

    top_supplier_share_rows = []

    for category, category_data in grouped:
        supplier_spend = (
            category_data.groupby("supplier_name")["annual_spend"]
            .sum()
            .sort_values(ascending=False)
        )

        top_supplier_share = (
            supplier_spend.iloc[0] / supplier_spend.sum() * 100
            if len(supplier_spend) > 0 and supplier_spend.sum() > 0
            else 0
        )

        tail_supplier_count = int(
            (
                    category_data.groupby("supplier_name")["annual_spend"]
                    .sum()
                    < 10000
            ).sum()
        )

        top_supplier_share_rows.append(
            {
                "category": category,
                "top_supplier_share_pct": top_supplier_share,
                "tail_supplier_count": tail_supplier_count,
            }
        )

    supplier_share_data = pd.DataFrame(top_supplier_share_rows)

    category_metrics = category_metrics.merge(
        supplier_share_data,
        on="category",
        how="left",
    )

    category_metrics["concentration_risk_flag"] = (
            category_metrics["top_supplier_share_pct"] >= 60
    )

    category_metrics["fragmentation_review_flag"] = (
            category_metrics["supplier_count"] >= 4
    )

    category_metrics["tail_spend_review_flag"] = (
            category_metrics["tail_supplier_count"] >= 3
    )

    return category_metrics.sort_values(
        by="total_category_spend",
        ascending=False,
    ).reset_index(drop=True)


def add_attention_scoring(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()

    if "annual_spend" not in data.columns:
        data["annual_spend"] = 0

    data["annual_spend"] = pd.to_numeric(
        data["annual_spend"],
        errors="coerce",
    ).fillna(0)

    if "prior_year_spend" in data.columns:
        prior_spend = pd.to_numeric(
            data["prior_year_spend"],
            errors="coerce",
        )

        data["spend_change_pct"] = (
                (data["annual_spend"] - prior_spend)
                / prior_spend.replace(0, pd.NA)
                * 100
        )
    else:
        data["spend_change_pct"] = pd.NA

    score = pd.Series(0, index=data.index, dtype="float")

    spend_threshold = data["annual_spend"].quantile(0.75)

    score += (data["annual_spend"] >= spend_threshold).astype(int) * 25

    if "spend_change_pct" in data.columns:
        score += (
                pd.to_numeric(data["spend_change_pct"], errors="coerce")
                .fillna(0)
                .gt(25)
                .astype(int)
                * 20
        )

    if "contract_status" in data.columns:
        no_contract = (
            data["contract_status"]
            .fillna("")
            .astype(str)
            .str.lower()
            .isin(["no contract", "not contracted", "none", "expired", "missing"])
        )
        score += no_contract.astype(int) * 20

    if "needs_classification_review" in data.columns:
        score += data["needs_classification_review"].fillna(False).astype(bool).astype(int) * 10

    if "supplier_criticality" in data.columns:
        criticality = data["supplier_criticality"].fillna("").astype(str).str.lower()
        score += criticality.isin(["high", "critical", "strategic"]).astype(int) * 15

    if "on_time_delivery_pct" in data.columns:
        otd = pd.to_numeric(data["on_time_delivery_pct"], errors="coerce")
        score += otd.lt(85).fillna(False).astype(int) * 10

    data["attention_score"] = score.clip(0, 100).round(0)

    data["attention_level"] = pd.cut(
        data["attention_score"],
        bins=[-1, 39, 69, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)

    return data


def prepare_supplier_data(
        raw_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, dict, dict]:
    mapped_data, mapping_report = map_columns(raw_data)

    supplier_data = clean_supplier_data(mapped_data)

    supplier_data = add_invoice_period_columns(supplier_data)

    date_summary = summarize_date_coverage(supplier_data)

    supplier_data = classify_spend_data(supplier_data)

    supplier_data["category"] = supplier_data["analysis_category"]

    aggregation_time_grain = (
        "month"
        if date_summary.get("has_date_column") and date_summary.get("valid_date_rows", 0) > 0
        else None
    )

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

    readiness_report = create_data_readiness_report(
        supplier_data,
        mapping_report=mapping_report,
    )

    if not readiness_report["minimum_required_ready"]:
        raise ValueError(
            "Uploaded file is missing required fields: "
            + ", ".join(readiness_report["missing_required_columns"])
        )

    supplier_data = ensure_optional_analysis_columns(supplier_data)

    supplier_data = add_attention_scoring(supplier_data)

    category_metrics = create_category_metrics(supplier_data)

    return (
        supplier_data,
        category_metrics,
        readiness_report,
        mapping_report,
        date_summary,
        aggregation_summary,
    )


def create_excel_download(dataframes: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, dataframe in dataframes.items():
            safe_sheet_name = sheet_name[:31]
            dataframe.to_excel(
                writer,
                sheet_name=safe_sheet_name,
                index=False,
            )

    return output.getvalue()


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------

def main():
    apply_custom_styles()

    st.markdown(
        """
        <div class="hero-card">
            <h2>Supplier Performance & Spend Intelligence Dashboard</h2>
            <p>
            Upload supplier spend or performance data to identify where money is going,
            classify suppliers, detect rationalization opportunities, estimate savings,
            and generate procurement next actions.
            </p>
            <p>
            <strong>Portfolio demo:</strong> Python, Streamlit, Pandas, Plotly, OpenPyXL, and Pytest.
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

    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx", "xls"],
    )

    st.sidebar.divider()

    st.sidebar.markdown(
        """
        **Recommended fields**
        - Supplier/vendor name
        - Description or category
        - Invoice amount / spend amount
        - Transaction date
        - Business unit
        - Region
        - Contract status
        - Payment terms
        """
    )

    if uploaded_file is None:
        st.info(
            "Upload a CSV or Excel supplier spend file to begin. "
            "Use the dummy test file we created earlier for your interview demo."
        )
        return

    try:
        raw_data = load_uploaded_data(uploaded_file)

        (
            supplier_data,
            category_metrics,
            readiness_report,
            mapping_report,
            date_summary,
            aggregation_summary,
        ) = prepare_supplier_data(raw_data)

    except Exception as error:
        st.error(f"Unable to process the dataset: {error}")
        return

    # Sidebar filters
    st.sidebar.divider()
    st.sidebar.header("Filters")

    available_categories = sorted(
        supplier_data["category"].dropna().astype(str).unique().tolist()
        if "category" in supplier_data.columns
        else []
    )

    selected_categories = st.sidebar.multiselect(
        "Categories",
        available_categories,
        default=available_categories,
    )

    available_attention_levels = sorted(
        supplier_data["attention_level"].dropna().astype(str).unique().tolist()
        if "attention_level" in supplier_data.columns
        else []
    )

    selected_attention_levels = st.sidebar.multiselect(
        "Attention levels",
        available_attention_levels,
        default=available_attention_levels,
    )

    filtered_supplier_data = supplier_data.copy()

    if selected_categories:
        filtered_supplier_data = filtered_supplier_data[
            filtered_supplier_data["category"].astype(str).isin(selected_categories)
        ]

    if selected_attention_levels and "attention_level" in filtered_supplier_data.columns:
        filtered_supplier_data = filtered_supplier_data[
            filtered_supplier_data["attention_level"].astype(str).isin(
                selected_attention_levels
            )
        ]

    filtered_category_metrics = create_category_metrics(filtered_supplier_data)

    classification_summary = summarize_classification_coverage(
        filtered_supplier_data
    )

    rationalization_opportunities = create_rationalization_opportunities(
        filtered_supplier_data
    )

    opportunity_summary = summarize_opportunity_pipeline(
        rationalization_opportunities
    )

    procurement_recommendations = create_procurement_recommendations(
        rationalization_opportunities
    )

    (
        executive_tab,
        spend_tab,
        rationalization_tab,
        savings_tab,
        methodology_tab,
    ) = st.tabs(
        [
            "1. Executive Summary",
            "2. Spend Analysis",
            "3. Rationalization Opportunities",
            "4. Savings Ideas",
            "5. Methodology & Data Readiness",
        ]
    )

    with executive_tab:
        st.subheader("Executive procurement summary")

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

        summary_columns = st.columns(5)

        summary_columns[0].metric(
            "Total spend analyzed",
            format_currency(total_spend),
        )

        summary_columns[1].metric(
            "Suppliers",
            format_number(supplier_count),
        )

        summary_columns[2].metric(
            "Categories",
            format_number(category_count),
        )

        summary_columns[3].metric(
            "Opportunities found",
            format_number(opportunity_summary["opportunity_count"]),
        )

        summary_columns[4].metric(
            "Est. savings range",
            f"{format_currency(opportunity_summary['estimated_savings_low'])} - {format_currency(opportunity_summary['estimated_savings_high'])}",
        )

        st.markdown(
            f"""
            <div class="insight-box">
            The highest-value opportunity area identified from the uploaded data is 
            <strong>{opportunity_summary["top_category"]}</strong>. Savings estimates are directional and should be validated
            through sourcing events, supplier negotiations, and contract review.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        left_chart_column, right_chart_column = st.columns(2)

        with left_chart_column:
            show_chart_or_message(
                create_top_categories_chart(filtered_supplier_data),
                "Category spend chart is not available for this file.",
            )

        with right_chart_column:
            show_chart_or_message(
                create_opportunity_savings_chart(rationalization_opportunities),
                "Savings opportunity chart is not available yet.",
            )

        st.divider()

        st.subheader("Top recommended actions")

        for index, recommendation in enumerate(
                procurement_recommendations[:5],
                start=1,
        ):
            st.markdown(
                f"""
                <div class="section-card">
                    <h4>{index}. {recommendation["recommendation"]}</h4>
                    <p><strong>Why this matters:</strong> {recommendation["why_it_matters"]}</p>
                    <p><strong>Expected benefit:</strong> {recommendation["expected_benefit"]}</p>
                    <p><strong>Next action:</strong> {recommendation["next_action"]}</p>
                    <p>
                        <strong>Estimated savings:</strong> {recommendation["estimated_savings_range"]} &nbsp; | &nbsp;
                        <strong>Priority:</strong> {recommendation["priority"]} &nbsp; | &nbsp;
                        <strong>Difficulty:</strong> {recommendation["difficulty"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with spend_tab:
        st.subheader("Spend analysis")

        spend_columns = st.columns(4)

        spend_columns[0].metric(
            "Total spend",
            format_currency(filtered_supplier_data["annual_spend"].sum()),
        )

        spend_columns[1].metric(
            "Suppliers",
            format_number(filtered_supplier_data["supplier_name"].nunique()),
        )

        spend_columns[2].metric(
            "Categories",
            format_number(filtered_supplier_data["category"].nunique()),
        )

        spend_columns[3].metric(
            "Avg spend / supplier",
            format_currency(
                filtered_supplier_data["annual_spend"].sum()
                / max(filtered_supplier_data["supplier_name"].nunique(), 1)
            ),
        )

        st.divider()

        chart_column_1, chart_column_2 = st.columns(2)

        with chart_column_1:
            show_chart_or_message(
                create_top_suppliers_chart(filtered_supplier_data),
                "Top supplier chart is not available for this file.",
            )

        with chart_column_2:
            show_chart_or_message(
                create_top_categories_chart(filtered_supplier_data),
                "Top category chart is not available for this file.",
            )

        st.divider()

        chart_column_3, chart_column_4 = st.columns(2)

        with chart_column_3:
            show_chart_or_message(
                create_supplier_count_by_category_chart(filtered_supplier_data),
                "Supplier count by category chart is not available for this file.",
            )

        with chart_column_4:
            show_chart_or_message(
                create_monthly_spend_trend_chart(filtered_supplier_data),
                "Spend trend chart is not available because usable date fields were not found.",
            )

        st.divider()

        chart_column_5, chart_column_6 = st.columns(2)

        with chart_column_5:
            show_chart_or_message(
                create_spend_by_region_chart(filtered_supplier_data),
                "Spend by region chart is not available for this file.",
            )

        with chart_column_6:
            show_chart_or_message(
                create_spend_by_business_unit_chart(filtered_supplier_data),
                "Spend by business unit chart is not available for this file.",
            )

        st.divider()

        with st.expander("View top suppliers table"):
            top_suppliers = (
                filtered_supplier_data.groupby("supplier_name", dropna=False)[
                    "annual_spend"
                ]
                .sum()
                .reset_index()
                .sort_values(by="annual_spend", ascending=False)
                .head(20)
            )

            st.dataframe(
                make_display_table(top_suppliers),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("View top categories table"):
            top_categories = (
                filtered_supplier_data.groupby("category", dropna=False)[
                    "annual_spend"
                ]
                .sum()
                .reset_index()
                .sort_values(by="annual_spend", ascending=False)
                .head(20)
            )

            st.dataframe(
                make_display_table(top_categories),
                use_container_width=True,
                hide_index=True,
            )

    with rationalization_tab:
        st.subheader("Supplier rationalization opportunities")

        st.markdown(
            """
            <div class="insight-box">
            This view identifies categories where supplier consolidation, tail-spend cleanup, or contract coverage review may create procurement value.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if rationalization_opportunities.empty:
            st.warning(
                "No rationalization opportunities were detected from the available data."
            )
        else:
            rationalization_columns = st.columns(4)

            rationalization_columns[0].metric(
                "Opportunities",
                format_number(len(rationalization_opportunities)),
            )

            rationalization_columns[1].metric(
                "High priority",
                format_number(opportunity_summary["high_priority_count"]),
            )

            rationalization_columns[2].metric(
                "Low savings estimate",
                format_currency(opportunity_summary["estimated_savings_low"]),
            )

            rationalization_columns[3].metric(
                "High savings estimate",
                format_currency(opportunity_summary["estimated_savings_high"]),
            )

            st.divider()

            left_opportunity_chart, right_opportunity_chart = st.columns(2)

            with left_opportunity_chart:
                show_chart_or_message(
                    create_opportunity_savings_chart(rationalization_opportunities),
                    "Savings by category chart is not available.",
                )

            with right_opportunity_chart:
                show_chart_or_message(
                    create_opportunity_type_chart(rationalization_opportunities),
                    "Opportunity mix chart is not available.",
                )

            st.divider()

            show_chart_or_message(
                create_priority_chart(rationalization_opportunities),
                "Priority chart is not available.",
            )

            st.divider()

            st.subheader("Opportunity detail")

            opportunity_display_columns = [
                "priority",
                "opportunity_type",
                "category",
                "current_suppliers",
                "total_spend",
                "suggested_supplier_reduction",
                "estimated_savings_range",
                "rationale",
                "next_action",
            ]

            available_opportunity_display_columns = [
                column
                for column in opportunity_display_columns
                if column in rationalization_opportunities.columns
            ]

            st.dataframe(
                make_display_table(
                    rationalization_opportunities[
                        available_opportunity_display_columns
                    ]
                ),
                use_container_width=True,
                hide_index=True,
            )

    with savings_tab:
        st.subheader("Savings ideas and procurement actions")

        savings_columns = st.columns(3)

        savings_columns[0].metric(
            "Directional savings low",
            format_currency(opportunity_summary["estimated_savings_low"]),
        )

        savings_columns[1].metric(
            "Directional savings high",
            format_currency(opportunity_summary["estimated_savings_high"]),
        )

        savings_columns[2].metric(
            "High-priority opportunities",
            format_number(opportunity_summary["high_priority_count"]),
        )

        st.markdown(
            """
            <div class="warning-box">
            Savings estimates are directional and should be validated through sourcing events,
            supplier negotiations, demand analysis, and contract review.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        left_savings_chart, right_savings_chart = st.columns(2)

        with left_savings_chart:
            show_chart_or_message(
                create_opportunity_savings_chart(rationalization_opportunities),
                "Savings chart is not available.",
            )

        with right_savings_chart:
            show_chart_or_message(
                create_priority_chart(rationalization_opportunities),
                "Priority chart is not available.",
            )

        st.divider()

        for index, recommendation in enumerate(
                procurement_recommendations,
                start=1,
        ):
            st.markdown(
                f"""
                <div class="section-card">
                    <h4>{index}. {recommendation["recommendation"]}</h4>
                    <p><strong>Why this matters:</strong> {recommendation["why_it_matters"]}</p>
                    <p><strong>Expected benefit:</strong> {recommendation["expected_benefit"]}</p>
                    <p><strong>Suggested next action:</strong> {recommendation["next_action"]}</p>
                    <p>
                        <strong>Savings range:</strong> {recommendation["estimated_savings_range"]} &nbsp; | &nbsp;
                        <strong>Priority:</strong> {recommendation["priority"]} &nbsp; | &nbsp;
                        <strong>Difficulty:</strong> {recommendation["difficulty"]}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with methodology_tab:
        st.subheader("What this project demonstrates")

        st.write(
            """
            This project is designed as a realistic supplier analytics workflow rather than a static dashboard.
            It handles messy uploaded files, maps common procurement column names, checks readiness before analysis,
            classifies spend using transparent rules, parses transaction dates, aggregates supplier/category rows,
            identifies rationalization opportunities, estimates savings directionally, and produces management-style recommendations.
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
            - **Rationalization logic:** identifies fragmented categories, tail spend, and contract coverage gaps.
            - **Savings estimation:** creates directional savings ranges based on simple procurement rules.
            - **Executive recommendations:** generates plain-English next actions for procurement teams.
            - **Downloadable outputs:** exports CSV and Excel outputs for further review.
            """
        )

        st.subheader("Savings methodology")

        st.markdown(
            """
            Savings estimates are directional and rule-based:

            - Fragmented category: typically 3%–12% depending on supplier count.
            - Tail spend cleanup: typically 3%–8%.
            - No contract coverage: typically 4%–10%.
            - Supplier consolidation: typically 4%–10%.

            These are not guaranteed savings. They are intended to prioritize where procurement teams should investigate further.
            """
        )

        st.divider()

        st.subheader("Downloads")

        excel_file = create_excel_download(
            {
                "Supplier Analysis": filtered_supplier_data,
                "Category Metrics": filtered_category_metrics,
                "Opportunities": rationalization_opportunities,
                "Column Mapping": mapping_report,
            }
        )

        st.download_button(
            label="Download Excel analysis workbook",
            data=excel_file,
            file_name="supplier_spend_intelligence_analysis.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.download_button(
            label="Download supplier analysis CSV",
            data=filtered_supplier_data.to_csv(index=False).encode("utf-8"),
            file_name="supplier_analysis.csv",
            mime="text/csv",
        )

        st.divider()

        st.subheader("Data readiness details")

        with st.expander("View data readiness diagnostics"):
            st.write(f"Readiness status: {readiness_report['analysis_status']}")
            st.write(f"Detected file type: {readiness_report['input_file_type']}")
            st.write(f"Mapped columns: {readiness_report['mapped_column_count']}")
            st.write(f"Unmapped columns: {readiness_report['unmapped_column_count']}")
            st.write(f"Date coverage: {date_summary['date_coverage_pct']}%")
            st.write(f"Aggregation grain: {aggregation_summary['time_grain']}")

            st.dataframe(
                make_display_table(readiness_report["analysis_capabilities"]),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("View classification sample"):
            classification_columns_to_show = [
                "supplier_name",
                "description",
                "annual_spend",
                "invoice_date",
                "invoice_month",
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

            st.dataframe(
                make_display_table(
                    filtered_supplier_data[
                        available_classification_columns
                    ].head(20)
                ),
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("View column mapping report"):
            st.dataframe(
                make_display_table(mapping_report),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    main()