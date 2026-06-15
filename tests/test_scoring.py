import pandas as pd

from src.scoring import add_supplier_attention_scores


def test_supplier_attention_score_columns_are_created():
    supplier_data = pd.DataFrame(
        {
            "category_spend_share_pct": [50.0, 10.0],
            "otd_change_pct_points": [-10.0, 2.0],
            "defect_rate_change_pct_points": [3.0, -1.0],
            "supplier_criticality": ["High", "Low"],
        }
    )

    scored_data = add_supplier_attention_scores(
        supplier_data
    )

    expected_columns = [
        "spend_exposure_score",
        "delivery_risk_score",
        "quality_risk_score",
        "criticality_score",
        "data_confidence_pct",
        "supplier_attention_score",
        "attention_level",
    ]

    for column in expected_columns:
        assert column in scored_data.columns


def test_high_risk_supplier_scores_higher_than_low_risk_supplier():
    supplier_data = pd.DataFrame(
        {
            "category_spend_share_pct": [50.0, 10.0],
            "otd_change_pct_points": [-10.0, 2.0],
            "defect_rate_change_pct_points": [3.0, -1.0],
            "supplier_criticality": ["High", "Low"],
        }
    )

    scored_data = add_supplier_attention_scores(
        supplier_data
    )

    high_risk_score = scored_data.loc[
        0,
        "supplier_attention_score",
    ]

    low_risk_score = scored_data.loc[
        1,
        "supplier_attention_score",
    ]

    assert high_risk_score > low_risk_score


def test_missing_data_reduces_data_confidence():
    supplier_data = pd.DataFrame(
        {
            "category_spend_share_pct": [50.0],
            "otd_change_pct_points": [None],
            "defect_rate_change_pct_points": [3.0],
            "supplier_criticality": ["High"],
        }
    )

    scored_data = add_supplier_attention_scores(
        supplier_data
    )

    assert scored_data.loc[0, "data_confidence_pct"] == 75.0