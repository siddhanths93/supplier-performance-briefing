import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_supplier_risk_scatter(
    supplier_data: pd.DataFrame,
) -> go.Figure:
    """
    Create a supplier risk-performance scatter plot.

    X-axis:
        Supplier attention score

    Y-axis:
        Annual spend

    Bubble size:
        Annual spend

    Bubble grouping:
        Attention level
    """
    chart_data = supplier_data.copy()

    chart_data = chart_data.dropna(
        subset=[
            "supplier_attention_score",
            "annual_spend",
        ]
    )

    figure = px.scatter(
        chart_data,
        x="supplier_attention_score",
        y="annual_spend",
        size="annual_spend",
        color="attention_level",
        hover_name="supplier_name",
        hover_data={
            "category": True,
            "annual_spend": ":,.0f",
            "supplier_attention_score": ":.1f",
            "data_confidence_pct": ":.1f",
        },
        labels={
            "supplier_attention_score": (
                "Supplier Attention Score"
            ),
            "annual_spend": "Annual Spend",
            "attention_level": "Attention Level",
        },
        title=(
            "Supplier Spend Exposure vs. "
            "Management Attention"
        ),
    )

    figure.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
        legend_title_text="Attention Level",
    )

    return figure


def create_category_concentration_chart(
    category_metrics: pd.DataFrame,
) -> go.Figure:
    """
    Compare top-supplier and top-three supplier shares by category.
    """
    chart_data = (
        category_metrics[
            [
                "category",
                "top_supplier_share_pct",
                "top_3_supplier_share_pct",
            ]
        ]
        .sort_values(
            by="top_3_supplier_share_pct",
            ascending=True,
        )
    )

    figure = go.Figure()

    figure.add_bar(
        name="Top Supplier Share",
        y=chart_data["category"],
        x=chart_data["top_supplier_share_pct"],
        orientation="h",
    )

    figure.add_bar(
        name="Top 3 Supplier Share",
        y=chart_data["category"],
        x=chart_data["top_3_supplier_share_pct"],
        orientation="h",
    )

    figure.update_layout(
        title="Supplier Concentration by Category",
        xaxis_title="Share of Category Spend (%)",
        yaxis_title="Category",
        barmode="group",
    )

    return figure