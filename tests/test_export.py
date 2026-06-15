import pandas as pd

from src.export import create_excel_briefing_workbook


def test_create_excel_briefing_workbook_returns_bytes():
    supplier_findings = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [4200000],
            "supplier_attention_score": [88.5],
            "attention_level": ["High"],
            "data_confidence_pct": [100.0],
            "observation": ["Apex has a review finding."],
            "implication": ["This may create operating risk."],
            "suggested_next_step": ["Review corrective actions."],
        }
    )

    category_findings = pd.DataFrame(
        {
            "category": ["Logistics"],
            "total_category_spend": [9000000],
            "observation": ["Logistics has concentration risk."],
            "implication": ["This may indicate supplier dependency."],
            "suggested_next_step": ["Review supplier dependency."],
        }
    )

    supplier_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [4200000],
            "category_spend_share_pct": [45.0],
            "spend_exposure_score": [90.0],
            "delivery_risk_score": [100.0],
            "quality_risk_score": [95.0],
            "criticality_score": [100.0],
            "supplier_attention_score": [88.5],
            "attention_level": ["High"],
            "data_confidence_pct": [100.0],
        }
    )

    category_metrics = pd.DataFrame(
        {
            "category": ["Logistics"],
            "total_category_spend": [9000000],
            "supplier_count": [8],
            "average_spend_per_supplier": [1125000],
            "median_spend_per_supplier": [750000],
            "top_supplier_share_pct": [45.0],
            "top_3_supplier_share_pct": [80.0],
            "top_5_supplier_share_pct": [95.0],
            "suppliers_to_80_pct_spend": [3],
            "hhi": [2800.0],
            "tail_supplier_count": [2],
            "tail_supplier_pct": [25.0],
            "tail_spend": [100000],
            "tail_spend_pct": [1.1],
            "concentration_risk_flag": [False],
            "fragmentation_review_flag": [False],
            "tail_spend_review_flag": [False],
        }
    )

    workbook_bytes = create_excel_briefing_workbook(
        supplier_findings=supplier_findings,
        category_findings=category_findings,
        supplier_data=supplier_data,
        category_metrics=category_metrics,
    )

    assert isinstance(workbook_bytes, bytes)
    assert len(workbook_bytes) > 0