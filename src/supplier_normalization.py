import re
from collections import defaultdict

import pandas as pd
from rapidfuzz import fuzz, process


KNOWN_SUPPLIER_ALIASES = {
    "ibm": "IBM",
    "i b m": "IBM",
    "international business machines": "IBM",
    "international business machines corporation": "IBM",

    "aws": "Amazon / AWS",
    "amazon web services": "Amazon / AWS",
    "amazon web services inc": "Amazon / AWS",
    "amazon": "Amazon / AWS",

    "microsoft": "Microsoft",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "msft": "Microsoft",
    "microsoft azure": "Microsoft",

    "google": "Google",
    "google cloud": "Google",
    "google llc": "Google",
    "alphabet": "Google",

    "dhl": "DHL",
    "dhl express": "DHL",
    "dhl global forwarding": "DHL",

    "fedex": "FedEx",
    "fedex corp": "FedEx",
    "fedex corporation": "FedEx",
    "federal express": "FedEx",

    "ups": "UPS",
    "united parcel service": "UPS",

    "oracle": "Oracle",
    "oracle corp": "Oracle",
    "oracle corporation": "Oracle",

    "sap": "SAP",
    "sap america": "SAP",

    "accenture": "Accenture",
    "accenture llp": "Accenture",

    "deloitte": "Deloitte",
    "deloitte consulting": "Deloitte",
    "deloitte touche": "Deloitte",
}


COMMON_SUFFIXES = [
    "inc",
    "incorporated",
    "llc",
    "l l c",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
    "plc",
    "lp",
    "llp",
    "gmbh",
    "sa",
    "ag",
    "bv",
    "pte",
    "holdings",
    "group",
    "services",
]


def clean_supplier_name(name) -> str:
    """
    Cleans supplier names for fuzzy matching.

    Example:
    'I.B.M. Corporation, LLC' -> 'i b m'
    """
    if pd.isna(name):
        return ""

    cleaned = str(name).lower().strip()

    # Replace punctuation with spaces
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)

    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = cleaned.split()

    # Remove common legal/entity suffixes
    words = [word for word in words if word not in COMMON_SUFFIXES]

    cleaned = " ".join(words).strip()

    return cleaned


def alias_lookup(original_name: str) -> str | None:
    """
    Applies known supplier alias dictionary before fuzzy matching.
    """
    cleaned = clean_supplier_name(original_name)

    if not cleaned:
        return None

    if cleaned in KNOWN_SUPPLIER_ALIASES:
        return KNOWN_SUPPLIER_ALIASES[cleaned]

    return None


def choose_canonical_supplier(original_names: list[str], spend_by_supplier: dict | None = None) -> str:
    """
    Chooses a canonical supplier name when no known alias exists.

    Preference:
    1. Highest-spend original supplier name
    2. Shortest readable supplier name
    """
    valid_names = [str(name).strip() for name in original_names if str(name).strip()]

    if not valid_names:
        return "Unknown Supplier"

    if spend_by_supplier:
        sorted_by_spend = sorted(
            valid_names,
            key=lambda supplier: spend_by_supplier.get(supplier, 0),
            reverse=True,
        )
        return sorted_by_spend[0]

    return sorted(valid_names, key=len)[0]


def build_supplier_normalization_mapping(
    df: pd.DataFrame,
    supplier_col: str = "supplier_name",
    spend_col: str = "annual_spend",
    threshold: int = 90,
) -> tuple[dict, pd.DataFrame]:
    """
    Builds a mapping from original supplier name to normalized supplier family.

    Returns:
    1. mapping dictionary
    2. supplier normalization summary dataframe
    """

    if supplier_col not in df.columns:
        return {}, pd.DataFrame()

    working = df.copy()

    working[supplier_col] = working[supplier_col].fillna("Unknown Supplier").astype(str)

    if spend_col in working.columns:
        working[spend_col] = pd.to_numeric(working[spend_col], errors="coerce").fillna(0)
    else:
        working[spend_col] = 0

    supplier_spend = (
        working.groupby(supplier_col, dropna=False)[spend_col]
        .sum()
        .to_dict()
    )

    original_suppliers = sorted(working[supplier_col].dropna().astype(str).unique())

    mapping = {}
    match_scores = {}
    match_methods = {}

    # Step 1: apply known aliases first
    unresolved = []

    for supplier in original_suppliers:
        alias = alias_lookup(supplier)

        if alias:
            mapping[supplier] = alias
            match_scores[supplier] = 100
            match_methods[supplier] = "Known alias"
        else:
            unresolved.append(supplier)

    # Step 2: fuzzy group unresolved supplier names
    cleaned_to_originals = defaultdict(list)

    for supplier in unresolved:
        cleaned = clean_supplier_name(supplier)

        if cleaned:
            cleaned_to_originals[cleaned].append(supplier)
        else:
            mapping[supplier] = "Unknown Supplier"
            match_scores[supplier] = 0
            match_methods[supplier] = "Missing supplier name"

    cleaned_names = list(cleaned_to_originals.keys())
    assigned_cleaned_names = set()

    for cleaned_name in cleaned_names:
        if cleaned_name in assigned_cleaned_names:
            continue

        # Find similar cleaned supplier names
        matches = process.extract(
            cleaned_name,
            cleaned_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
            limit=None,
        )

        matched_cleaned_names = [match[0] for match in matches]
        matched_scores = [match[1] for match in matches]

        for matched_name in matched_cleaned_names:
            assigned_cleaned_names.add(matched_name)

        original_group = []

        for matched_name in matched_cleaned_names:
            original_group.extend(cleaned_to_originals[matched_name])

        canonical = choose_canonical_supplier(
            original_group,
            spend_by_supplier=supplier_spend,
        )

        average_score = round(sum(matched_scores) / len(matched_scores), 1) if matched_scores else 0

        for original_supplier in original_group:
            mapping[original_supplier] = canonical
            match_scores[original_supplier] = average_score
            match_methods[original_supplier] = "Fuzzy match" if len(original_group) > 1 else "No close match"

    # Step 3: create summary table
    summary_rows = []

    grouped_variants = defaultdict(list)

    for original_supplier, normalized_supplier in mapping.items():
        grouped_variants[normalized_supplier].append(original_supplier)

    for normalized_supplier, variants in grouped_variants.items():
        total_spend = sum(supplier_spend.get(variant, 0) for variant in variants)
        scores = [match_scores.get(variant, 0) for variant in variants]
        methods = sorted(set(match_methods.get(variant, "Unknown") for variant in variants))

        summary_rows.append(
            {
                "normalized_supplier_name": normalized_supplier,
                "original_supplier_variants": ", ".join(sorted(variants)),
                "variant_count": len(variants),
                "total_spend": total_spend,
                "average_match_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "match_method": ", ".join(methods),
            }
        )

    summary = pd.DataFrame(summary_rows)

    if not summary.empty:
        summary = summary.sort_values(
            by=["total_spend", "variant_count"],
            ascending=[False, False],
        )

    return mapping, summary


def apply_supplier_normalization(
    df: pd.DataFrame,
    supplier_col: str = "supplier_name",
    spend_col: str = "annual_spend",
    threshold: int = 90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Adds normalized supplier columns to the dataset.

    Output columns added:
    - original_supplier_name
    - normalized_supplier_name
    """

    if df.empty or supplier_col not in df.columns:
        return df, pd.DataFrame()

    normalized_df = df.copy()

    mapping, summary = build_supplier_normalization_mapping(
        normalized_df,
        supplier_col=supplier_col,
        spend_col=spend_col,
        threshold=threshold,
    )

    normalized_df["original_supplier_name"] = normalized_df[supplier_col]
    normalized_df["normalized_supplier_name"] = normalized_df[supplier_col].map(mapping)

    normalized_df["normalized_supplier_name"] = normalized_df["normalized_supplier_name"].fillna(
        normalized_df[supplier_col]
    )

    return normalized_df, summary