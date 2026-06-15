import pandas as pd

from src.recommendations import create_supplier_findings


def test_create_supplier_findings_returns_expected_columns():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [4200000],
            "category_spend_share_pct": [45.0],
            "supplier_attention_score": [88.5],
            "attention_level": ["High"],
            "data_confidence_pct": [100.0],
            "high_spend_exposure_flag": [True],
            "delivery_deterioration_flag": [True],
            "quality_deterioration_flag": [True],
            "missing_performance_data_flag": [False],
            "otd_change_pct_points": [-13.0],
            "defect_rate_change_pct_points": [2.7],
            "supplier_criticality": ["High"],
        }
    )

    findings = create_supplier_findings(
        supplier_data,
        top_n=1,
    )

    expected_columns = [
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

    for column in expected_columns:
        assert column in findings.columns


def test_supplier_finding_contains_evidence_based_language():
    supplier_data = pd.DataFrame(
        {
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [4200000],
            "category_spend_share_pct": [45.0],
            "supplier_attention_score": [88.5],
            "attention_level": ["High"],
            "data_confidence_pct": [100.0],
            "high_spend_exposure_flag": [True],
            "delivery_deterioration_flag": [True],
            "quality_deterioration_flag": [True],
            "missing_performance_data_flag": [False],
            "otd_change_pct_points": [-13.0],
            "defect_rate_change_pct_points": [2.7],
            "supplier_criticality": ["High"],
        }
    )

    findings = create_supplier_findings(
        supplier_data,
        top_n=1,
    )

    observation = findings.loc[0, "observation"]

    assert "Apex Freight Solutions" in observation
    assert "annual spend" in observation
    assert "on-time delivery declined" in observation
    assert "defect rate increased" in observation