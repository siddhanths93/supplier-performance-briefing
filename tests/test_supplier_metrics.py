import pytest
import pandas as pd

from src.supplier_metrics import calculate_supplier_metrics


def test_total_spend_share_adds_to_100_percent():
    supplier_data = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2", "S3"],
            "supplier_name": ["A", "B", "C"],
            "category": ["Logistics", "Logistics", "IT"],
            "annual_spend": [500, 300, 200],
            "prior_year_spend": [400, 300, 100],
            "on_time_delivery_pct": [90, 80, 95],
            "prior_year_otd_pct": [95, 80, 90],
            "defect_rate_pct": [3, 2, 1],
            "prior_year_defect_rate_pct": [2, 2, 2],
        }
    )

    metrics = calculate_supplier_metrics(
        supplier_data
    )

    assert metrics["total_spend_share_pct"].sum() == pytest.approx(100.0)


def test_category_spend_share_adds_to_100_percent_by_category():
    supplier_data = pd.DataFrame(
        {
            "supplier_id": ["S1", "S2", "S3"],
            "supplier_name": ["A", "B", "C"],
            "category": ["Logistics", "Logistics", "IT"],
            "annual_spend": [500, 300, 200],
            "prior_year_spend": [400, 300, 100],
            "on_time_delivery_pct": [90, 80, 95],
            "prior_year_otd_pct": [95, 80, 90],
            "defect_rate_pct": [3, 2, 1],
            "prior_year_defect_rate_pct": [2, 2, 2],
        }
    )

    metrics = calculate_supplier_metrics(
        supplier_data
    )

    category_share_totals = metrics.groupby(
        "category"
    )["category_spend_share_pct"].sum()

    assert category_share_totals["Logistics"] == pytest.approx(100.0)
    assert category_share_totals["IT"] == pytest.approx(100.0)


def test_spend_change_and_performance_change_are_calculated_correctly():
    supplier_data = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [1200],
            "prior_year_spend": [1000],
            "on_time_delivery_pct": [81],
            "prior_year_otd_pct": [94],
            "defect_rate_pct": [4.8],
            "prior_year_defect_rate_pct": [2.1],
        }
    )

    metrics = calculate_supplier_metrics(
        supplier_data
    )

    assert metrics.loc[0, "spend_change_amount"] == 200
    assert metrics.loc[0, "spend_change_pct"] == pytest.approx(20.0)
    assert metrics.loc[0, "otd_change_pct_points"] == pytest.approx(-13.0)
    assert metrics.loc[0, "defect_rate_change_pct_points"] == pytest.approx(2.7)


def test_supplier_flags_are_triggered_correctly():
    supplier_data = pd.DataFrame(
        {
            "supplier_id": ["S1"],
            "supplier_name": ["Apex Freight Solutions"],
            "category": ["Logistics"],
            "annual_spend": [1200],
            "prior_year_spend": [1000],
            "on_time_delivery_pct": [81],
            "prior_year_otd_pct": [94],
            "defect_rate_pct": [4.8],
            "prior_year_defect_rate_pct": [2.1],
        }
    )

    metrics = calculate_supplier_metrics(
        supplier_data
    )

    assert metrics.loc[0, "delivery_deterioration_flag"] == True
    assert metrics.loc[0, "quality_deterioration_flag"] == True
    assert metrics.loc[0, "high_spend_exposure_flag"] == True