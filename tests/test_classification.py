import pandas as pd

from src.classification import (
    classify_spend_data,
    determine_confidence,
    normalize_text,
    split_keywords,
    summarize_classification_coverage,
)


def test_split_keywords_returns_clean_keyword_list():
    keywords = split_keywords(
        " cloud ; AWS ; hosting "
    )

    assert keywords == [
        "cloud",
        "aws",
        "hosting",
    ]


def test_normalize_text_removes_symbols_and_extra_spaces():
    normalized = normalize_text(
        " Amazon-Web Services, Inc. "
    )

    assert normalized == "amazon web services inc"


def test_determine_confidence_from_score():
    assert determine_confidence(120) == "High"
    assert determine_confidence(50) == "Medium"
    assert determine_confidence(10) == "Low"
    assert determine_confidence(0) == "Low"


def test_supplier_keyword_classification_returns_high_confidence():
    data = pd.DataFrame(
        {
            "supplier_name": ["DHL Express"],
            "description": ["International shipment"],
            "annual_spend": [1000],
        }
    )

    classified_data = classify_spend_data(data)

    assert classified_data.loc[0, "taxonomy_level_1"] == "Logistics"
    assert classified_data.loc[0, "taxonomy_level_2"] == "Freight & Parcel"
    assert classified_data.loc[0, "classification_confidence"] == "High"
    assert bool(classified_data.loc[0, "needs_classification_review"]) is False


def test_broad_uploaded_category_does_not_force_specific_subcategory():
    data = pd.DataFrame(
        {
            "supplier_name": ["Unknown Supplier"],
            "category": ["Marketing"],
            "description": ["General marketing expense"],
            "annual_spend": [500],
        }
    )

    classified_data = classify_spend_data(data)

    assert classified_data.loc[0, "analysis_category"] == "Marketing"
    assert classified_data.loc[0, "classification_source"] == "uploaded_category_fallback"
    assert bool(classified_data.loc[0, "needs_classification_review"]) is True



def test_description_keyword_classification_returns_valid_confidence():
    data = pd.DataFrame(
        {
            "supplier_name": ["Unknown Supplier"],
            "description": ["Monthly cloud hosting charges"],
            "annual_spend": [500],
        }
    )

    classified_data = classify_spend_data(data)

    assert classified_data.loc[0, "taxonomy_level_1"] == "IT"
    assert classified_data.loc[0, "taxonomy_level_2"] == "Cloud Infrastructure"
    assert classified_data.loc[0, "classification_confidence"] in [
        "Medium",
        "High",
    ]
    assert classified_data.loc[0, "classification_score"] >= 40


def test_specific_phrase_can_outscore_generic_keyword():
    data = pd.DataFrame(
        {
            "supplier_name": ["Unknown Supplier"],
            "description": ["Need office chair replacement"],
            "annual_spend": [500],
        }
    )

    classified_data = classify_spend_data(data)

    assert classified_data.loc[0, "taxonomy_level_2"] == "Furniture"


def test_exclude_keyword_prevents_bad_cloud_match():
    data = pd.DataFrame(
        {
            "supplier_name": ["Unknown Supplier"],
            "description": ["Cloud conference sponsorship booth"],
            "annual_spend": [500],
        }
    )

    classified_data = classify_spend_data(data)

    assert classified_data.loc[0, "taxonomy_level_2"] != "Cloud Infrastructure"


def test_uploaded_category_fallback_when_no_keyword_matches():
    data = pd.DataFrame(
        {
            "supplier_name": ["Unknown Supplier"],
            "category": ["Special Projects"],
            "description": ["Unusual service"],
            "annual_spend": [500],
        }
    )

    classified_data = classify_spend_data(data)

    assert classified_data.loc[0, "analysis_category"] == "Special Projects"
    assert classified_data.loc[0, "classification_source"] == "uploaded_category_fallback"
    assert bool(classified_data.loc[0, "needs_classification_review"]) is True


def test_unclassified_when_no_text_or_category_available():
    data = pd.DataFrame(
        {
            "supplier_name": [None],
            "description": [None],
            "annual_spend": [500],
        }
    )

    classified_data = classify_spend_data(data)

    assert classified_data.loc[0, "analysis_category"] == "Unclassified"
    assert bool(classified_data.loc[0, "needs_classification_review"]) is True


def test_summarize_classification_coverage():
    data = pd.DataFrame(
        {
            "supplier_name": [
                "DHL Express",
                None,
            ],
            "description": [
                "Freight shipment",
                None,
            ],
            "annual_spend": [
                1000,
                500,
            ],
        }
    )

    classified_data = classify_spend_data(data)

    summary = summarize_classification_coverage(
        classified_data
    )

    assert summary["classified_rows"] == 1
    assert summary["unclassified_rows"] == 1
    assert summary["review_required_rows"] == 1
    assert summary["classification_coverage_pct"] == 50.0