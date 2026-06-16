from pathlib import Path
import re

import pandas as pd


DEFAULT_TAXONOMY_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "reference"
    / "spend_taxonomy.csv"
)

SUPPLIER_KEYWORD_SCORE = 100
TEXT_KEYWORD_SCORE = 40
UPLOADED_CATEGORY_SCORE = 30
LONG_PHRASE_BONUS = 10
EXCLUSION_PENALTY = -10_000


def split_keywords(keyword_text) -> list[str]:
    """
    Split a semicolon-separated keyword string into clean keywords.
    """
    if pd.isna(keyword_text):
        return []

    return [
        keyword.strip().lower()
        for keyword in str(keyword_text).split(";")
        if keyword.strip()
    ]


def normalize_text(value) -> str:
    """
    Normalize text for matching.
    """
    if pd.isna(value):
        return ""

    normalized = str(value).lower()

    normalized = re.sub(
        r"[^a-z0-9]+",
        " ",
        normalized,
    )

    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    return normalized


def load_spend_taxonomy(
    taxonomy_path: str | Path = DEFAULT_TAXONOMY_PATH,
) -> pd.DataFrame:
    """
    Load the built-in spend taxonomy reference file.
    """
    taxonomy_path = Path(taxonomy_path)

    if not taxonomy_path.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found: {taxonomy_path}"
        )

    taxonomy = pd.read_csv(taxonomy_path)

    required_columns = {
        "taxonomy_level_1",
        "taxonomy_level_2",
        "taxonomy_code",
        "description",
        "keywords",
        "supplier_keywords",
    }

    missing_columns = required_columns - set(taxonomy.columns)

    if missing_columns:
        raise ValueError(
            "Taxonomy file is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    if "exclude_keywords" not in taxonomy.columns:
        taxonomy["exclude_keywords"] = ""

    return taxonomy


def build_search_text(row: pd.Series) -> str:
    """
    Build searchable text from available supplier, description, and category fields.
    """
    text_parts = []

    for column in [
        "supplier_name",
        "description",
        "category",
    ]:
        if column in row.index and not pd.isna(row[column]):
            text_parts.append(str(row[column]))

    return normalize_text(" ".join(text_parts))


def keyword_in_text(
    keyword: str,
    search_text: str,
) -> bool:
    """
    Check whether a normalized keyword appears in normalized search text.
    """
    normalized_keyword = normalize_text(keyword)

    if not normalized_keyword:
        return False

    return normalized_keyword in search_text


def phrase_bonus(keyword: str) -> int:
    """
    Give a small bonus to longer, more specific keyword phrases.
    """
    normalized_keyword = normalize_text(keyword)

    if len(normalized_keyword.split()) >= 2:
        return LONG_PHRASE_BONUS

    return 0


def score_taxonomy_row(
    row: pd.Series,
    taxonomy_row: pd.Series,
) -> dict[str, object]:
    """
    Score how well one taxonomy row matches one spend row.
    """
    search_text = build_search_text(row)

    if not search_text:
        return {
            "score": 0,
            "matched_reasons": [],
            "excluded": False,
        }

    score = 0
    matched_reasons = []

    exclude_keywords = split_keywords(
        taxonomy_row.get("exclude_keywords", "")
    )

    for keyword in exclude_keywords:
        if keyword_in_text(keyword, search_text):
            return {
                "score": EXCLUSION_PENALTY,
                "matched_reasons": [
                    f"Excluded by keyword: {keyword}"
                ],
                "excluded": True,
            }

    supplier_keywords = split_keywords(
        taxonomy_row["supplier_keywords"]
    )

    for keyword in supplier_keywords:
        if keyword_in_text(keyword, search_text):
            keyword_score = (
                SUPPLIER_KEYWORD_SCORE
                + phrase_bonus(keyword)
            )
            score += keyword_score
            matched_reasons.append(
                f"supplier keyword '{keyword}' +{keyword_score}"
            )

    text_keywords = split_keywords(
        taxonomy_row["keywords"]
    )

    for keyword in text_keywords:
        if keyword_in_text(keyword, search_text):
            keyword_score = (
                TEXT_KEYWORD_SCORE
                + phrase_bonus(keyword)
            )
            score += keyword_score
            matched_reasons.append(
                f"text keyword '{keyword}' +{keyword_score}"
            )

    if "category" in row.index and not pd.isna(row["category"]):
        uploaded_category = normalize_text(row["category"])

        taxonomy_level_2 = normalize_text(
            taxonomy_row["taxonomy_level_2"]
        )

        # Only use uploaded category as a scoring signal when it matches
        # the more specific taxonomy level 2.
        #
        # Do NOT score broad level 1 matches such as:
        # Marketing -> Advertising
        # Professional Services -> Consulting
        # Facilities -> Maintenance
        #
        # Those are too broad and should fall back to the uploaded category
        # unless supplier or description keywords provide more evidence.
        if uploaded_category == taxonomy_level_2:
            score += UPLOADED_CATEGORY_SCORE
            matched_reasons.append(
                f"uploaded category level 2 match +{UPLOADED_CATEGORY_SCORE}"
            )

    return {
        "score": score,
        "matched_reasons": matched_reasons,
        "excluded": False,
    }


def determine_confidence(score: int) -> str:
    """
    Convert classification score into confidence label.
    """
    if score >= 100:
        return "High"

    if score >= 40:
        return "Medium"

    if score > 0:
        return "Low"

    return "Low"


def get_uploaded_category_value(
        row: pd.Series,
) -> str:
    """
    Return uploaded category value if available.
    """
    if "category" not in row.index:
        return ""

    if pd.isna(row["category"]):
        return ""

    return str(row["category"]).strip()


def create_user_friendly_reason(
        matched_reasons: list[str],
) -> str:
    """
    Convert internal scoring reasons into user-facing explanation text.
    """
    if not matched_reasons:
        return "No matching rule explanation available."

    cleaned_reasons = []

    for reason in matched_reasons:
        cleaned_reason = reason

        cleaned_reason = re.sub(
            r"\s\+\d+",
            "",
            cleaned_reason,
        )

        cleaned_reason = cleaned_reason.replace(
            "supplier keyword",
            "Matched supplier keyword",
        )

        cleaned_reason = cleaned_reason.replace(
            "text keyword",
            "Matched description/category keyword",
        )

        cleaned_reason = cleaned_reason.replace(
            "uploaded category level 2 match",
            "Matched uploaded category to taxonomy subcategory",
        )

        cleaned_reasons.append(cleaned_reason)

    return "; ".join(cleaned_reasons)



def classify_row(
    row: pd.Series,
    taxonomy: pd.DataFrame,
) -> dict[str, str | bool | int]:
    """
    Classify one supplier/spend row using scored taxonomy rules.
    """
    search_text = build_search_text(row)

    if not search_text.strip():
        return {
            "taxonomy_level_1": "Unclassified",
            "taxonomy_level_2": "Unclassified",
            "taxonomy_code": "UNCLASSIFIED",
            "analysis_category": "Unclassified",
            "classification_source": "no_text_available",
            "classification_confidence": "Low",
            "classification_score": 0,
            "classification_reason": (
                "No supplier, category, or description text available."
            ),
            "needs_classification_review": True,
        }

    scored_matches = []

    for _, taxonomy_row in taxonomy.iterrows():
        score_result = score_taxonomy_row(
            row,
            taxonomy_row,
        )

        scored_matches.append(
            {
                "taxonomy_row": taxonomy_row,
                "score": score_result["score"],
                "matched_reasons": score_result["matched_reasons"],
                "excluded": score_result["excluded"],
            }
        )

    valid_matches = [
        match
        for match in scored_matches
        if match["score"] > 0
    ]

    if valid_matches:
        best_match = max(
            valid_matches,
            key=lambda match: match["score"],
        )

        taxonomy_row = best_match["taxonomy_row"]
        score = int(best_match["score"])
        confidence = determine_confidence(score)

        return {
            "taxonomy_level_1": taxonomy_row["taxonomy_level_1"],
            "taxonomy_level_2": taxonomy_row["taxonomy_level_2"],
            "taxonomy_code": taxonomy_row["taxonomy_code"],
            "analysis_category": taxonomy_row["taxonomy_level_2"],
            "classification_source": "scored_builtin_rules",
            "classification_confidence": confidence,
            "classification_score": score,
            "classification_reason": create_user_friendly_reason(
                best_match["matched_reasons"]
            ),
            "needs_classification_review": (
                confidence != "High"
            ),
        }

    category_value = get_uploaded_category_value(row)

    if category_value:
        return {
            "taxonomy_level_1": category_value,
            "taxonomy_level_2": category_value,
            "taxonomy_code": "UPLOADED_CATEGORY",
            "analysis_category": category_value,
            "classification_source": "uploaded_category_fallback",
            "classification_confidence": "Low",
            "classification_score": 0,
            "classification_reason": (
                "No supplier or description keyword matched; used uploaded category as-is."
            ),
            "needs_classification_review": True,
        }

    return {
        "taxonomy_level_1": "Unclassified",
        "taxonomy_level_2": "Unclassified",
        "taxonomy_code": "UNCLASSIFIED",
        "analysis_category": "Unclassified",
        "classification_source": "unclassified",
        "classification_confidence": "Low",
        "classification_score": 0,
        "classification_reason": (
            "No taxonomy keyword or uploaded category matched."
        ),
        "needs_classification_review": True,
    }


def classify_spend_data(
    data: pd.DataFrame,
    taxonomy: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Classify supplier/spend rows into the built-in spend taxonomy.
    """
    data = data.copy()

    if taxonomy is None:
        taxonomy = load_spend_taxonomy()

    classifications = [
        classify_row(row, taxonomy)
        for _, row in data.iterrows()
    ]

    classification_data = pd.DataFrame(
        classifications,
        index=data.index,
    )

    classified_data = pd.concat(
        [
            data,
            classification_data,
        ],
        axis=1,
    )

    return classified_data


def summarize_classification_coverage(
    data: pd.DataFrame,
) -> dict[str, int | float]:
    """
    Summarize classification coverage and review needs.
    """
    if "needs_classification_review" not in data.columns:
        return {
            "classified_rows": 0,
            "unclassified_rows": 0,
            "review_required_rows": 0,
            "classification_coverage_pct": 0.0,
        }

    total_rows = len(data)

    review_required_rows = int(
        data["needs_classification_review"].sum()
    )

    unclassified_rows = int(
        (
            data["analysis_category"]
            == "Unclassified"
        ).sum()
    )

    classified_rows = total_rows - unclassified_rows

    coverage_pct = (
        classified_rows / total_rows * 100
        if total_rows > 0
        else 0.0
    )

    return {
        "classified_rows": classified_rows,
        "unclassified_rows": unclassified_rows,
        "review_required_rows": review_required_rows,
        "classification_coverage_pct": round(
            coverage_pct,
            1,
        ),
    }