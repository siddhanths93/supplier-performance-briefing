import pandas as pd


def create_supplier_briefing_export(
    supplier_findings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a clean supplier findings export table.
    """
    if supplier_findings.empty:
        return pd.DataFrame()

    export_data = supplier_findings.copy()

    export_data = export_data[
        [
            "supplier_name",
            "category",
            "annual_spend",
            "supplier_attention_score",
            "attention_level",
            "data_confidence_pct",
            "observation",
            "implication",
            "suggested_next_step",
        ]
    ]

    export_data = export_data.rename(
        columns={
            "supplier_name": "Supplier",
            "category": "Category",
            "annual_spend": "Annual Spend",
            "supplier_attention_score": "Attention Score",
            "attention_level": "Attention Level",
            "data_confidence_pct": "Data Confidence %",
            "observation": "Observation",
            "implication": "Implication",
            "suggested_next_step": "Suggested Next Step",
        }
    )

    return export_data


def create_category_briefing_export(
    category_findings: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a clean category findings export table.
    """
    if category_findings.empty:
        return pd.DataFrame()

    export_data = category_findings.copy()

    export_data = export_data[
        [
            "category",
            "total_category_spend",
            "observation",
            "implication",
            "suggested_next_step",
        ]
    ]

    export_data = export_data.rename(
        columns={
            "category": "Category",
            "total_category_spend": "Total Category Spend",
            "observation": "Observation",
            "implication": "Implication",
            "suggested_next_step": "Suggested Next Step",
        }
    )

    return export_data