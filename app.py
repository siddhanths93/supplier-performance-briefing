import re
from collections import defaultdict
from io import BytesIO

import pandas as pd
import plotly.express as px
import streamlit as st
from rapidfuzz import fuzz, process


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Supplier Performance Diagnostic Workbench",
    layout="wide",
)


# ============================================================
# GLOBAL STYLING
# ============================================================

def apply_global_styles():
    st.markdown(
        """
        <style>
        .main {
            background-color: #f8fafc;
        }

        .hero-card {
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #312e81 100%);
            padding: 2.2rem;
            border-radius: 1.4rem;
            color: white;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 45px rgba(15, 23, 42, 0.25);
        }

        .hero-label {
            text-transform: uppercase;
            letter-spacing: 0.32em;
            font-size: 0.78rem;
            color: #bfdbfe;
            margin-bottom: 0.9rem;
            font-weight: 700;
        }

        .hero-title {
            color: white;
            margin-bottom: 0.75rem;
            font-size: 2.25rem;
            font-weight: 850;
            line-height: 1.1;
        }

        .hero-subtitle {
            color: #dbeafe;
            font-size: 1.02rem;
            line-height: 1.65;
            max-width: 1100px;
        }

        .section-card {
            background-color: white;
            padding: 1.2rem 1.3rem;
            border-radius: 1rem;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.06);
            margin-bottom: 1rem;
        }

        .callout-card {
            background: #eff6ff;
            border: 1px solid #bfdbfe;
            padding: 1.1rem 1.2rem;
            border-radius: 1rem;
            color: #1e3a8a;
            margin-bottom: 1rem;
        }

        .warning-card {
            background: #fffbeb;
            border: 1px solid #fde68a;
            padding: 1.1rem 1.2rem;
            border-radius: 1rem;
            color: #92400e;
            margin-bottom: 1rem;
        }

        .risk-high {
            color: #991b1b;
            font-weight: 700;
        }

        .risk-medium {
            color: #92400e;
            font-weight: 700;
        }

        .risk-low {
            color: #166534;
            font-weight: 700;
        }

        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 1.6rem;
            font-weight: 800;
        }

        div[data-testid="stMetricLabel"] {
            color: #475569;
            font-weight: 600;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FORMAT HELPERS
# ============================================================

def format_currency(value):
    try:
        value = float(value)
    except Exception:
        return "Not available"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def format_percent(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "Not available"


def format_number(value):
    try:
        return f"{float(value):,.0f}"
    except Exception:
        return "Not available"


def prettify_column_name(column_name):
    return str(column_name).replace("_", " ").title()


def make_display_table(df):
    display = df.copy()
    display.columns = [prettify_column_name(col) for col in display.columns]
    return display


def apply_chart_layout(fig, height=420):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=65, b=30),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(size=12),
        title=dict(font=dict(size=18)),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    return fig


def show_chart_or_message(fig, message, key):
    if fig is None:
        st.info(message)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


# ============================================================
# COLUMN MAPPING
# ============================================================

COLUMN_ALIASES = {
    "supplier_name": [
        "supplier_name", "supplier name", "supplier", "vendor", "vendor name",
        "vendor_name", "supplier_nm", "suppliername", "supplier family",
    ],
    "annual_spend": [
        "annual_spend", "annual spend", "spend", "spend amount", "spend_amount",
        "invoice amount", "invoice_amount", "totalcost", "total cost", "total_cost",
        "total amount", "total_amount", "amount", "purchase amount", "purchase_amount",
        "spend_amount_usd", "spend amount usd", "usd spend", "extended cost",
    ],
    "category": [
        "category", "spend category", "category_l1", "category l1",
        "commodity", "commodity group", "item category", "procurement category",
    ],
    "description": [
        "description", "item description", "item_description", "itemname",
        "item name", "item_name", "product", "product name", "transaction description",
    ],
    "business_unit": [
        "business_unit", "business unit", "bu", "department", "dept", "cost center",
        "cost_center", "function",
    ],
    "region": [
        "region", "geo", "geography", "market", "territory",
    ],
    "country": [
        "country", "supplier country", "country_name", "location country",
    ],
    "contract_status": [
        "contract_status", "contract status", "contract", "agreement status",
        "contract coverage", "contract_coverage",
    ],
    "supplier_criticality": [
        "supplier_criticality", "supplier criticality", "criticality",
        "business criticality", "critical supplier", "risk criticality",
    ],
    "on_time_delivery": [
        "on_time_delivery", "on time delivery", "otd", "otd_percent",
        "otd percentage", "delivery performance", "delivery score",
    ],
    "prior_year_otd": [
        "prior_year_otd", "prior year otd", "previous year otd", "last year otd",
        "prior_otd", "py otd",
    ],
    "defect_rate": [
        "defect_rate", "defect rate", "quality defect rate", "defects",
        "quality score", "quality_defect_rate",
    ],
    "prior_year_defect_rate": [
        "prior_year_defect_rate", "prior year defect rate", "previous year defect rate",
        "prior defect rate", "py defect rate",
    ],
    "lead_time_days": [
        "lead_time_days", "lead time days", "lead time", "leadtime", "avg lead time",
        "average lead time",
    ],
    "invoice_count": [
        "invoice_count", "invoice count", "invoices", "number of invoices",
    ],
    "po_count": [
        "po_count", "po count", "purchase orders", "purchase order count",
    ],
    "buyer": [
        "buyer", "buyer name", "buyer_name", "purchaser", "requester",
    ],
    "payment_terms": [
        "payment_terms", "payment terms", "terms", "supplier payment terms",
    ],
    "invoice_date": [
        "invoice_date", "invoice date", "purchase_date", "purchase date",
        "purchasedate", "transaction date", "transaction_date", "date",
    ],
    "quantity": [
        "quantity", "qty", "order quantity", "order_quantity",
    ],
    "unit_price": [
        "unit_price", "unit price", "unitprice", "price", "unit cost", "unit_cost",
    ],
}


def normalize_column_name(column):
    normalized = str(column).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_column_mapping(df):
    original_columns = list(df.columns)
    normalized_to_original = {normalize_column_name(col): col for col in original_columns}

    mapping = {}

    for standard_col, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            normalized_alias = normalize_column_name(alias)
            if normalized_alias in normalized_to_original:
                mapping[normalized_to_original[normalized_alias]] = standard_col
                break

    return mapping


def standardize_columns(df):
    data = df.copy()
    mapping = build_column_mapping(data)
    data = data.rename(columns=mapping)

    # If annual spend is missing but quantity and unit price exist, calculate it.
    if "annual_spend" not in data.columns and {"quantity", "unit_price"}.issubset(data.columns):
        data["quantity"] = pd.to_numeric(data["quantity"], errors="coerce").fillna(0)
        data["unit_price"] = pd.to_numeric(data["unit_price"], errors="coerce").fillna(0)
        data["annual_spend"] = data["quantity"] * data["unit_price"]

    # If required analytical columns are missing, degrade gracefully.
    if "supplier_name" not in data.columns:
        data["supplier_name"] = "Unknown Supplier"

    if "annual_spend" not in data.columns:
        data["annual_spend"] = 0

    if "category" not in data.columns:
        data["category"] = "Unclassified"

    data["supplier_name"] = data["supplier_name"].fillna("Unknown Supplier").astype(str)
    data["category"] = data["category"].fillna("Unclassified").astype(str)

    data["annual_spend"] = (
        data["annual_spend"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    data["annual_spend"] = pd.to_numeric(data["annual_spend"], errors="coerce").fillna(0)

    numeric_columns = [
        "on_time_delivery",
        "prior_year_otd",
        "defect_rate",
        "prior_year_defect_rate",
        "lead_time_days",
        "invoice_count",
        "po_count",
        "quantity",
        "unit_price",
    ]

    for column in numeric_columns:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Convert decimal percentages to percentage-point scale if needed.
    for column in ["on_time_delivery", "prior_year_otd", "defect_rate", "prior_year_defect_rate"]:
        if column in data.columns:
            valid = data[column].dropna()
            if not valid.empty and valid.max() <= 1:
                data[column] = data[column] * 100

    if "invoice_date" in data.columns:
        data["invoice_date"] = pd.to_datetime(data["invoice_date"], errors="coerce")

    return data, mapping


# ============================================================
# FILE LOADING
# ============================================================

def score_sheet_for_spend_data(df):
    if df.empty:
        return 0

    columns = [normalize_column_name(col) for col in df.columns]
    score = 0

    supplier_terms = ["supplier", "vendor"]
    spend_terms = ["spend", "amount", "cost", "totalcost", "invoice"]
    category_terms = ["category", "commodity"]
    date_terms = ["date", "purchase", "invoice"]

    if any(any(term in col for term in supplier_terms) for col in columns):
        score += 4
    if any(any(term in col for term in spend_terms) for col in columns):
        score += 4
    if any(any(term in col for term in category_terms) for col in columns):
        score += 2
    if any(any(term in col for term in date_terms) for col in columns):
        score += 1

    score += min(len(df), 1000) / 1000

    return score


def load_uploaded_data(uploaded_file):
    if uploaded_file is None:
        return None, None

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        return df, "CSV Upload"

    if file_name.endswith((".xlsx", ".xls")):
        excel_file = pd.ExcelFile(uploaded_file)
        best_sheet_name = None
        best_sheet_df = None
        best_score = -1

        for sheet_name in excel_file.sheet_names:
            candidate = pd.read_excel(excel_file, sheet_name=sheet_name)
            score = score_sheet_for_spend_data(candidate)

            if score > best_score:
                best_score = score
                best_sheet_name = sheet_name
                best_sheet_df = candidate

        return best_sheet_df, f"Excel Sheet: {best_sheet_name}"

    raise ValueError("Unsupported file type. Please upload CSV or Excel.")


# ============================================================
# DEMO DATA
# ============================================================

def build_demo_dataset(demo_type):
    rows = []

    if demo_type == "Logistics supplier performance":
        suppliers = [
            ("DHL Express", "DHL Express", "Logistics", "High", "Active", 1_200_000, 81, 94, 1.2, 1.0, 6),
            ("D.H.L.", "DHL Express", "Logistics", "High", "Active", 260_000, 83, 93, 1.1, 0.9, 6),
            ("FedEx Corp", "FedEx", "Logistics", "Medium", "Active", 850_000, 92, 91, 0.8, 0.9, 5),
            ("UPS", "UPS", "Logistics", "Medium", "Expiring Soon", 740_000, 89, 94, 1.9, 1.2, 7),
            ("Regional Freight LLC", "Regional Freight", "Logistics", "Low", "No Contract", 160_000, 86, 88, 2.2, 2.0, 9),
        ]
    elif demo_type == "IT services supplier performance":
        suppliers = [
            ("Microsoft Corp", "Microsoft", "IT Services", "High", "Active", 2_400_000, 98, 98, 0.1, 0.2, 3),
            ("MSFT", "Microsoft", "IT Services", "High", "Active", 340_000, 97, 97, 0.1, 0.2, 3),
            ("Amazon Web Services", "Amazon / AWS", "Cloud", "High", "Active", 1_950_000, 96, 96, 0.3, 0.2, 2),
            ("AWS", "Amazon / AWS", "Cloud", "High", "Active", 540_000, 96, 97, 0.3, 0.2, 2),
            ("Local IT Support Co", "Local IT Support", "IT Services", "Medium", "Unknown", 420_000, 84, 91, 2.8, 1.4, 8),
        ]
    elif demo_type == "MRO supplier performance":
        suppliers = [
            ("Critical MRO Supplier", "Critical MRO Supplier", "MRO", "High", "Unknown", 900_000, 82, 91, 3.8, 2.2, 14),
            ("Grainger", "Grainger", "MRO", "Medium", "Active", 720_000, 95, 94, 0.9, 1.0, 5),
            ("Fastenal Co", "Fastenal", "MRO", "Medium", "Active", 510_000, 93, 94, 1.1, 1.0, 5),
            ("Local Bearings LLC", "Local Bearings", "MRO", "Low", "No Contract", 85_000, 88, 89, 2.1, 1.9, 10),
            ("Industrial Parts Inc", "Industrial Parts", "MRO", "Low", "Expired", 140_000, 86, 90, 2.6, 1.7, 12),
        ]
    elif demo_type == "Professional services supplier performance":
        suppliers = [
            ("Deloitte Consulting", "Deloitte", "Professional Services", "High", "Active", 1_800_000, 97, 97, 0.0, 0.0, 4),
            ("Accenture LLP", "Accenture", "Professional Services", "High", "Active", 1_300_000, 94, 96, 0.0, 0.0, 5),
            ("Local Staffing LLC", "Local Staffing", "Professional Services", "Medium", "Expired", 520_000, 88, 92, 0.4, 0.1, 9),
            ("Boutique Advisory Co", "Boutique Advisory", "Professional Services", "Low", "Unknown", 180_000, 91, 91, 0.0, 0.0, 6),
            ("Contractor Services Inc", "Contractor Services", "Professional Services", "Low", "No Contract", 110_000, 87, 89, 0.2, 0.1, 7),
        ]
    else:
        suppliers = [
            ("DHL Express", "DHL Express", "Logistics", "High", "Active", 1_200_000, 81, 94, 1.2, 1.0, 6),
            ("Microsoft Corp", "Microsoft", "IT Services", "High", "Active", 2_400_000, 98, 98, 0.1, 0.2, 3),
            ("Critical MRO Supplier", "Critical MRO Supplier", "MRO", "High", "Unknown", 900_000, 82, 91, 3.8, 2.2, 14),
            ("Local HVAC Repair Co", "Local HVAC", "Facilities", "Medium", "No Contract", 420_000, 88, 91, 2.0, 1.5, 11),
            ("Office Depot", "Office Depot", "Office Supplies", "Low", "Active", 310_000, 94, 94, 0.8, 0.8, 5),
            ("Staples Inc", "Staples", "Office Supplies", "Low", "Active", 260_000, 95, 94, 0.7, 0.8, 5),
            ("Local Office Supply", "Local Office Supply", "Office Supplies", "Low", "Unknown", 95_000, 90, 91, 1.5, 1.2, 7),
        ]

    business_units = ["Operations", "Corporate", "Manufacturing", "Field Services", "IT"]
    regions = ["North America", "South", "West", "Midwest", "East"]
    buyers = ["S. Patel", "A. Johnson", "M. Chen", "R. Singh", "L. Garcia"]

    transaction_id = 1

    for supplier_name, supplier_family, category, criticality, contract_status, total_spend, otd, prior_otd, defect, prior_defect, lead_time in suppliers:
        parts = 6
        for i in range(parts):
            spend = total_spend / parts
            rows.append(
                {
                    "transaction_id": f"TXN-{transaction_id:05d}",
                    "supplier_name": supplier_name,
                    "annual_spend": spend,
                    "category": category,
                    "business_unit": business_units[i % len(business_units)],
                    "region": regions[i % len(regions)],
                    "country": "United States",
                    "contract_status": contract_status,
                    "supplier_criticality": criticality,
                    "on_time_delivery": otd + (i % 3 - 1),
                    "prior_year_otd": prior_otd,
                    "defect_rate": defect,
                    "prior_year_defect_rate": prior_defect,
                    "lead_time_days": lead_time + (i % 2),
                    "invoice_count": 5 + i,
                    "po_count": 2 + i,
                    "buyer": buyers[i % len(buyers)],
                    "payment_terms": "Net 60" if i % 2 == 0 else "Net 45",
                    "invoice_date": pd.Timestamp("2026-01-01") + pd.DateOffset(months=i),
                    "description": f"{category} services from {supplier_family}",
                }
            )
            transaction_id += 1

    return pd.DataFrame(rows)


# ============================================================
# SUPPLIER FUZZY NORMALIZATION
# ============================================================

KNOWN_SUPPLIER_ALIASES = {
    "ibm": "IBM",
    "i b m": "IBM",
    "international business machines": "IBM",
    "international business machines corporation": "IBM",
    "aws": "Amazon / AWS",
    "amazon web services": "Amazon / AWS",
    "amazon": "Amazon / AWS",
    "microsoft": "Microsoft",
    "microsoft corp": "Microsoft",
    "microsoft corporation": "Microsoft",
    "msft": "Microsoft",
    "microsoft azure": "Microsoft",
    "google": "Google",
    "google cloud": "Google",
    "alphabet": "Google",
    "dhl": "DHL",
    "dhl express": "DHL",
    "dhl global forwarding": "DHL",
    "fedex": "FedEx",
    "fedex corp": "FedEx",
    "federal express": "FedEx",
    "ups": "UPS",
    "united parcel service": "UPS",
    "oracle": "Oracle",
    "sap": "SAP",
    "grainger": "Grainger",
    "fastenal": "Fastenal",
    "staples": "Staples",
    "staples inc": "Staples",
    "office depot": "Office Depot",
}

COMMON_SUFFIXES = [
    "inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation",
    "co", "company", "plc", "lp", "llp", "gmbh", "services", "service",
    "group", "holdings",
]


def clean_supplier_name(name):
    if pd.isna(name):
        return ""

    cleaned = str(name).lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = [word for word in cleaned.split() if word not in COMMON_SUFFIXES]
    return " ".join(words).strip()


def alias_lookup(name):
    cleaned = clean_supplier_name(name)
    return KNOWN_SUPPLIER_ALIASES.get(cleaned)


def choose_canonical_supplier(original_names, spend_by_supplier):
    valid = [str(name).strip() for name in original_names if str(name).strip()]

    if not valid:
        return "Unknown Supplier"

    sorted_by_spend = sorted(
        valid,
        key=lambda supplier: spend_by_supplier.get(supplier, 0),
        reverse=True,
    )

    return sorted_by_spend[0]


def apply_supplier_normalization(df, threshold=90):
    if df.empty or "supplier_name" not in df.columns:
        df["original_supplier_name"] = "Unknown Supplier"
        df["normalized_supplier_name"] = "Unknown Supplier"
        return df, pd.DataFrame()

    data = df.copy()
    data["original_supplier_name"] = data["supplier_name"].fillna("Unknown Supplier").astype(str)

    spend_by_supplier = (
        data.groupby("original_supplier_name")["annual_spend"]
        .sum()
        .to_dict()
    )

    original_suppliers = sorted(data["original_supplier_name"].unique())

    mapping = {}
    match_scores = {}
    match_methods = {}

    unresolved = []

    for supplier in original_suppliers:
        alias = alias_lookup(supplier)
        if alias:
            mapping[supplier] = alias
            match_scores[supplier] = 100
            match_methods[supplier] = "Known alias"
        else:
            unresolved.append(supplier)

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
    assigned = set()

    for cleaned in cleaned_names:
        if cleaned in assigned:
            continue

        matches = process.extract(
            cleaned,
            cleaned_names,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=threshold,
            limit=None,
        )

        matched_cleaned_names = [match[0] for match in matches]
        matched_scores = [match[1] for match in matches]

        for match_name in matched_cleaned_names:
            assigned.add(match_name)

        original_group = []
        for match_name in matched_cleaned_names:
            original_group.extend(cleaned_to_originals[match_name])

        canonical = choose_canonical_supplier(original_group, spend_by_supplier)
        average_score = round(sum(matched_scores) / len(matched_scores), 1) if matched_scores else 0

        for supplier in original_group:
            mapping[supplier] = canonical
            match_scores[supplier] = average_score
            match_methods[supplier] = "Fuzzy match" if len(original_group) > 1 else "No close match"

    data["normalized_supplier_name"] = data["original_supplier_name"].map(mapping).fillna(data["original_supplier_name"])

    rows = []

    for family, family_df in data.groupby("normalized_supplier_name", dropna=False):
        variants = sorted(family_df["original_supplier_name"].dropna().astype(str).unique())
        scores = [match_scores.get(variant, 0) for variant in variants]
        methods = sorted(set(match_methods.get(variant, "Unknown") for variant in variants))

        rows.append(
            {
                "normalized_supplier_name": family,
                "original_supplier_variants": ", ".join(variants),
                "variant_count": len(variants),
                "total_spend": family_df["annual_spend"].sum(),
                "average_match_score": round(sum(scores) / len(scores), 1) if scores else 0,
                "match_method": ", ".join(methods),
            }
        )

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary = summary.sort_values(["total_spend", "variant_count"], ascending=[False, False])

    return data, summary


# ============================================================
# SUPPLIER DIAGNOSTIC ENGINE
# ============================================================

def first_available_column(df, columns):
    for col in columns:
        if col in df.columns:
            return col
    return None


def safe_mode(series, default="Not available"):
    valid = series.dropna().astype(str)
    if valid.empty:
        return default
    return valid.mode().iloc[0]


def calculate_supplier_summary(results):
    if results.empty:
        return pd.DataFrame()

    supplier_col = "normalized_supplier_name" if "normalized_supplier_name" in results.columns else "supplier_name"

    group_cols = [supplier_col]

    aggregation = {
        "annual_spend": "sum",
        "supplier_name": pd.Series.nunique,
    }

    optional_mean_cols = [
        "on_time_delivery",
        "prior_year_otd",
        "defect_rate",
        "prior_year_defect_rate",
        "lead_time_days",
        "invoice_count",
        "po_count",
    ]

    for col in optional_mean_cols:
        if col in results.columns:
            aggregation[col] = "mean"

    supplier_summary = (
        results.groupby(group_cols, dropna=False)
        .agg(aggregation)
        .reset_index()
        .rename(
            columns={
                supplier_col: "supplier",
                "supplier_name": "original_supplier_count",
            }
        )
    )

    for col in ["category", "business_unit", "region", "country", "contract_status", "supplier_criticality", "buyer", "payment_terms"]:
        if col in results.columns:
            values = results.groupby(supplier_col)[col].agg(lambda x: safe_mode(x)).reset_index()
            values = values.rename(columns={supplier_col: "supplier"})
            supplier_summary = supplier_summary.merge(values, on="supplier", how="left")

    return supplier_summary


def classify_contract_risk(status, spend=0, high_spend_threshold=0, contract_data_available=True):
    """
    Classifies contract risk only when contract data is actually available.

    Important consulting logic:
    - Missing contract_status column = Not Evaluated
    - Present but Unknown contract_status value = Data Validation Required
    - Expired / No Contract = High
    - Active = Low
    """

    if not contract_data_available:
        return "Not Evaluated"

    status_text = str(status).strip().lower()

    if status_text in ["active", "active contract", "contracted"]:
        return "Low"

    if status_text in ["expiring soon", "expiring", "renewal due"]:
        return "Medium"

    if status_text in ["expired", "no contract", "none", "missing", "not contracted"]:
        return "High"

    if status_text in ["unknown", "not available", "n/a", "na", ""]:
        return "Data Validation Required"

    return "Data Validation Required"


def get_performance_flags(row):
    flags = []

    otd = row.get("on_time_delivery")
    prior_otd = row.get("prior_year_otd")
    defect = row.get("defect_rate")
    prior_defect = row.get("prior_year_defect_rate")
    lead_time = row.get("lead_time_days")

    if pd.notna(otd) and pd.notna(prior_otd):
        decline = float(prior_otd) - float(otd)
        if decline > 5:
            flags.append(
                {
                    "metric": "On-time delivery",
                    "current_value": format_percent(otd),
                    "prior_value": format_percent(prior_otd),
                    "change": f"-{decline:.1f} pts",
                    "threshold": "Decline greater than 5 pts",
                    "business_implication": "Potential service reliability issue requiring supplier review.",
                }
            )

    if pd.notna(defect) and pd.notna(prior_defect):
        increase = float(defect) - float(prior_defect)
        if increase > 1:
            flags.append(
                {
                    "metric": "Defect rate",
                    "current_value": format_percent(defect),
                    "prior_value": format_percent(prior_defect),
                    "change": f"+{increase:.1f} pts",
                    "threshold": "Increase greater than 1 pt",
                    "business_implication": "Potential quality deterioration requiring root-cause review.",
                }
            )

    if pd.notna(otd) and float(otd) < 90:
        flags.append(
            {
                "metric": "On-time delivery",
                "current_value": format_percent(otd),
                "prior_value": "Not required",
                "change": "Below threshold",
                "threshold": "Below 90%",
                "business_implication": "Supplier may be underperforming on delivery reliability.",
            }
        )

    if pd.notna(defect) and float(defect) > 3:
        flags.append(
            {
                "metric": "Defect rate",
                "current_value": format_percent(defect),
                "prior_value": "Not required",
                "change": "Above threshold",
                "threshold": "Above 3%",
                "business_implication": "Supplier may require corrective quality action.",
            }
        )

    if pd.notna(lead_time) and float(lead_time) > 10:
        flags.append(
            {
                "metric": "Lead time",
                "current_value": f"{float(lead_time):.1f} days",
                "prior_value": "Not required",
                "change": "Above threshold",
                "threshold": "Above 10 days",
                "business_implication": "Long lead time may create continuity or planning risk.",
            }
        )

    return flags


def assign_supplier_archetypes(supplier_summary):
    if supplier_summary.empty:
        return supplier_summary

    data = supplier_summary.copy()

    high_spend_threshold = data["annual_spend"].quantile(0.75) if len(data) > 1 else data["annual_spend"].max()
    medium_spend_threshold = data["annual_spend"].quantile(0.50) if len(data) > 1 else data["annual_spend"].median()

    contract_data_available = "contract_status" in data.columns

    archetypes = []
    actions = []
    why_flags = []
    priorities = []
    contract_risks = []
    risk_flags = []

    for _, row in data.iterrows():
        spend = float(row.get("annual_spend", 0))
        criticality = str(row.get("supplier_criticality", "Unknown")).strip().lower()

        if contract_data_available:
            contract_status = row.get("contract_status", "Unknown")
        else:
            contract_status = "Not Provided"

        is_high_spend = spend >= high_spend_threshold and spend > 0
        is_medium_spend = spend >= medium_spend_threshold and spend > 0
        is_low_spend = spend < medium_spend_threshold
        is_high_criticality = criticality in ["high", "critical", "business critical", "strategic"]
        is_medium_criticality = criticality in ["medium", "moderate"]

        contract_risk = classify_contract_risk(
            contract_status,
            spend,
            high_spend_threshold,
            contract_data_available=contract_data_available,
        )

        flags = get_performance_flags(row)

        poor_performance = len(flags) > 0
        strong_performance = True

        if pd.notna(row.get("on_time_delivery")) and float(row.get("on_time_delivery")) < 95:
            strong_performance = False

        if pd.notna(row.get("defect_rate")) and float(row.get("defect_rate")) > 1:
            strong_performance = False

        reasons = []

        if is_high_spend:
            reasons.append(f"high spend supplier at {format_currency(spend)}")
        elif is_medium_spend:
            reasons.append(f"moderate spend supplier at {format_currency(spend)}")
        else:
            reasons.append(f"lower spend supplier at {format_currency(spend)}")

        if is_high_criticality:
            reasons.append("supplier criticality is High")
        elif is_medium_criticality:
            reasons.append("supplier criticality is Medium")

        if contract_data_available:
            if contract_risk in ["High", "Medium", "Data Validation Required"]:
                reasons.append(f"contract status requires review: {contract_status}")
        else:
            reasons.append("contract status was not provided, so contract risk was not evaluated")

        for flag in flags[:2]:
            reasons.append(f"{flag['metric']} issue: {flag['change']}")

        # Important: contract risk should not drive archetypes unless contract data exists.
        has_real_contract_gap = contract_data_available and contract_risk in ["High", "Medium"]
        has_contract_validation_need = contract_data_available and contract_risk == "Data Validation Required"

        if is_high_criticality and poor_performance:
            archetype = "Alternate Source Required"
            action = "Identify backup suppliers and reduce dependency risk."
            priority = "High"
            risk_flag = True

        elif (is_high_spend or is_high_criticality) and poor_performance:
            archetype = "Executive Escalation"
            action = "Escalate in QBR, require corrective action plan, and assess commercial remedies."
            priority = "High"
            risk_flag = True

        elif (is_high_spend or is_medium_spend) and has_real_contract_gap:
            archetype = "Contract Review Priority"
            action = "Validate contract coverage, renewal timing, and commercial protection."
            priority = "High" if is_high_spend else "Medium"
            risk_flag = True

        elif (is_high_spend or is_medium_spend) and has_contract_validation_need:
            archetype = "Data Validation Required"
            action = "Validate contract metadata before assigning contract risk."
            priority = "Medium"
            risk_flag = False

        elif is_high_spend and strong_performance and contract_risk in ["Low", "Not Evaluated"] and (is_high_criticality or is_medium_criticality):
            archetype = "Strategic Partner"
            action = "Maintain relationship, deepen collaboration, and review value-add opportunities."
            priority = "Medium"
            risk_flag = False

        elif poor_performance:
            archetype = "Watchlist Supplier"
            action = "Monitor closely, request improvement plan, and review again in 30–60 days."
            priority = "Medium"
            risk_flag = True

        elif is_low_spend and has_real_contract_gap:
            archetype = "Rationalization Candidate"
            action = "Consider consolidation, replacement, or migration to a preferred supplier."
            priority = "Medium"
            risk_flag = True

        else:
            archetype = "Maintain / Monitor"
            action = "Continue monitoring through standard supplier management process."
            priority = "Low"
            risk_flag = False

        archetypes.append(archetype)
        actions.append(action)
        priorities.append(priority)
        contract_risks.append(contract_risk)
        risk_flags.append(risk_flag)
        why_flags.append("Flagged because " + ", ".join(reasons) + ".")

    data["action_archetype"] = archetypes
    data["recommended_action"] = actions
    data["priority"] = priorities
    data["contract_risk"] = contract_risks
    data["is_spend_at_risk"] = risk_flags
    data["why_flagged"] = why_flags

    return data

def calculate_spend_at_risk(supplier_diagnostics):
    if supplier_diagnostics.empty:
        return {}

    total_spend = supplier_diagnostics["annual_spend"].sum()

    risk_archetypes = [
        "Executive Escalation",
        "Watchlist Supplier",
        "Contract Review Priority",
        "Rationalization Candidate",
        "Alternate Source Required",
    ]

    spend_at_risk = supplier_diagnostics[
        supplier_diagnostics["action_archetype"].isin(risk_archetypes)
    ]["annual_spend"].sum()

    if "supplier_criticality" in supplier_diagnostics.columns:
        high_criticality_spend = supplier_diagnostics[
            supplier_diagnostics["supplier_criticality"]
            .astype(str)
            .str.lower()
            .isin(["high", "critical", "business critical", "strategic"])
        ]["annual_spend"].sum()
    else:
        high_criticality_spend = 0

    # Only count contract gap exposure if contract status was actually provided.
    if "contract_status" in supplier_diagnostics.columns and "contract_risk" in supplier_diagnostics.columns:
        contract_gap_spend = supplier_diagnostics[
            supplier_diagnostics["contract_risk"].isin(["High", "Medium"])
        ]["annual_spend"].sum()
    else:
        contract_gap_spend = None

    declining_spend = 0

    for _, row in supplier_diagnostics.iterrows():
        if len(get_performance_flags(row)) > 0:
            declining_spend += float(row.get("annual_spend", 0))

    return {
        "total_spend": total_spend,
        "spend_at_risk": spend_at_risk,
        "spend_at_risk_pct": (spend_at_risk / total_spend * 100) if total_spend > 0 else 0,
        "high_criticality_spend": high_criticality_spend,
        "contract_gap_spend": contract_gap_spend,
        "declining_performance_spend": declining_spend,
    }

# ============================================================
# OPPORTUNITY AND PIPELINE ENGINE
# ============================================================

def estimate_savings_range(opportunity_type, spend, supplier_count=0):
    if opportunity_type == "Fragmented category":
        if supplier_count >= 8:
            low, high = 0.07, 0.12
        elif supplier_count >= 5:
            low, high = 0.05, 0.09
        else:
            low, high = 0.03, 0.06
    elif opportunity_type == "Tail spend cleanup":
        low, high = 0.03, 0.08
    elif opportunity_type == "Contract coverage review":
        low, high = 0.04, 0.10
    else:
        low, high = 0.02, 0.05

    return spend * low, spend * high


def create_sourcing_opportunities(results, supplier_diagnostics):
    opportunities = []

    if results.empty:
        return pd.DataFrame()

    category_summary = (
        results.groupby("category", dropna=False)
        .agg(
            category_spend=("annual_spend", "sum"),
            supplier_count=("normalized_supplier_name", "nunique")
            if "normalized_supplier_name" in results.columns
            else ("supplier_name", "nunique"),
        )
        .reset_index()
    )

    for _, row in category_summary.iterrows():
        category = row["category"]
        spend = float(row["category_spend"])
        supplier_count = int(row["supplier_count"])

        if supplier_count >= 5:
            low, high = estimate_savings_range("Fragmented category", spend, supplier_count)
            opportunities.append(
                {
                    "opportunity_type": "Fragmented category",
                    "category": category,
                    "opportunity_action": f"Supplier consolidation review for {category}",
                    "value_low": low,
                    "value_high": high,
                    "value_display": f"{format_currency(low)} - {format_currency(high)}",
                    "priority": "High" if spend >= category_summary["category_spend"].quantile(0.75) else "Medium",
                    "confidence": "Medium",
                    "complexity": "Medium",
                    "why_flagged": f"Category has {supplier_count} suppliers and {format_currency(spend)} in spend, indicating possible fragmentation.",
                }
            )

    supplier_spend = (
        results.groupby("normalized_supplier_name" if "normalized_supplier_name" in results.columns else "supplier_name")
        .agg(spend=("annual_spend", "sum"))
        .reset_index()
    )

    if not supplier_spend.empty:
        tail_threshold = supplier_spend["spend"].quantile(0.25)
        tail_suppliers = supplier_spend[supplier_spend["spend"] <= tail_threshold]

        if len(tail_suppliers) >= 5:
            tail_spend = tail_suppliers["spend"].sum()
            low, high = estimate_savings_range("Tail spend cleanup", tail_spend)
            opportunities.append(
                {
                    "opportunity_type": "Tail spend cleanup",
                    "category": "Cross-category tail spend",
                    "opportunity_action": "Tail spend cleanup for low-spend suppliers",
                    "value_low": low,
                    "value_high": high,
                    "value_display": f"{format_currency(low)} - {format_currency(high)}",
                    "priority": "Medium",
                    "confidence": "Medium",
                    "complexity": "Low",
                    "why_flagged": f"{len(tail_suppliers)} suppliers sit in the low-spend segment, representing {format_currency(tail_spend)} in spend.",
                }
            )

    if not supplier_diagnostics.empty and "contract_risk" in supplier_diagnostics.columns:
        contract_gap = supplier_diagnostics[supplier_diagnostics["contract_risk"].isin(["High", "Medium"])]

        if not contract_gap.empty:
            contract_gap_by_category = (
                contract_gap.groupby("category", dropna=False)["annual_spend"]
                .sum()
                .reset_index()
                .sort_values("annual_spend", ascending=False)
                .head(5)
            )

            for _, row in contract_gap_by_category.iterrows():
                spend = float(row["annual_spend"])
                low, high = estimate_savings_range("Contract coverage review", spend)
                opportunities.append(
                    {
                        "opportunity_type": "Contract coverage review",
                        "category": row["category"],
                        "opportunity_action": f"Contract coverage review for {row['category']}",
                        "value_low": low,
                        "value_high": high,
                        "value_display": f"{format_currency(low)} - {format_currency(high)}",
                        "priority": "High",
                        "confidence": "Medium",
                        "complexity": "Low",
                        "why_flagged": f"{format_currency(spend)} in spend has High or Medium contract coverage risk.",
                    }
                )

    return pd.DataFrame(opportunities)


def create_action_pipeline(results, supplier_diagnostics, opportunities):
    rows = []

    if supplier_diagnostics is not None and not supplier_diagnostics.empty:
        action_archetypes_for_pipeline = [
            "Executive Escalation",
            "Watchlist Supplier",
            "Contract Review Priority",
            "Rationalization Candidate",
            "Alternate Source Required",
        ]

        pipeline_suppliers = supplier_diagnostics[
            supplier_diagnostics["action_archetype"].isin(action_archetypes_for_pipeline)
        ].copy()

        for _, row in pipeline_suppliers.iterrows():
            supplier = row.get("supplier", "Unknown Supplier")
            archetype = row.get("action_archetype", "Supplier review")
            spend = float(row.get("annual_spend", 0))

            if archetype == "Executive Escalation":
                action = f"Corrective action plan for {supplier}"
                owner = "Category Manager / Supplier Relationship Lead"
                next_step = "Prepare QBR escalation package and request corrective action plan."
                validation = "Confirm SLA language, delivery/quality root cause, and business impact."
                complexity = "Medium"

            elif archetype == "Alternate Source Required":
                action = f"Alternate source assessment for {supplier}"
                owner = "Category Manager"
                next_step = "Identify backup suppliers and qualification requirements."
                validation = "Confirm supplier criticality, switching constraints, and operational dependency."
                complexity = "High"

            elif archetype == "Contract Review Priority":
                action = f"Contract coverage review for {supplier}"
                owner = "Sourcing / Contracts Lead"
                next_step = "Validate contract status, renewal timing, pricing terms, and commercial protections."
                validation = "Confirm whether spend is unmanaged or contract metadata is incomplete."
                complexity = "Low"

            elif archetype == "Rationalization Candidate":
                action = f"Supplier rationalization review for {supplier}"
                owner = "Procurement Ops"
                next_step = "Assess whether spend can migrate to preferred suppliers or buying channels."
                validation = "Confirm supplier is non-strategic and substitutable."
                complexity = "Low"

            else:
                action = f"Watchlist review for {supplier}"
                owner = "Supplier Manager"
                next_step = "Monitor performance and request an improvement plan if trend continues."
                validation = "Confirm whether issues are recurring, isolated, or caused by internal demand."
                complexity = "Low"

            rows.append(
                {
                    "opportunity_action": action,
                    "supplier_or_category": supplier,
                    "category": row.get("category", "Unclassified"),
                    "action_archetype": archetype,
                    "value_exposure": spend,
                    "value_exposure_display": format_currency(spend),
                    "priority": row.get("priority", "Medium"),
                    "confidence": "High" if archetype in ["Executive Escalation", "Alternate Source Required"] else "Medium",
                    "complexity": complexity,
                    "recommended_owner": owner,
                    "next_step": next_step,
                    "validation_required": validation,
                    "status": "New",
                    "why_flagged": row.get("why_flagged", "Rule-based diagnostic flag."),
                }
            )

    if opportunities is not None and not opportunities.empty:
        for _, row in opportunities.iterrows():
            rows.append(
                {
                    "opportunity_action": row.get("opportunity_action", "Sourcing opportunity review"),
                    "supplier_or_category": row.get("category", "Unclassified"),
                    "category": row.get("category", "Unclassified"),
                    "action_archetype": row.get("opportunity_type", "Sourcing opportunity"),
                    "value_exposure": float(row.get("value_high", 0)),
                    "value_exposure_display": row.get("value_display", "Not estimated"),
                    "priority": row.get("priority", "Medium"),
                    "confidence": row.get("confidence", "Medium"),
                    "complexity": row.get("complexity", "Medium"),
                    "recommended_owner": "Procurement Ops / Category Manager",
                    "next_step": "Validate sourcing feasibility and confirm business case.",
                    "validation_required": "Confirm supplier substitutability, demand requirements, and category owner alignment.",
                    "status": "New",
                    "why_flagged": row.get("why_flagged", "Rule-based sourcing opportunity."),
                }
            )

    pipeline = pd.DataFrame(rows)

    if pipeline.empty:
        return pipeline

    priority_order = {"High": 1, "Medium": 2, "Low": 3}
    pipeline["priority_sort"] = pipeline["priority"].map(priority_order).fillna(99)

    pipeline = pipeline.sort_values(
        ["priority_sort", "value_exposure"],
        ascending=[True, False],
    ).drop(columns=["priority_sort"])

    return pipeline.reset_index(drop=True)


# ============================================================
# DATA READINESS
# ============================================================

def calculate_data_readiness(results, supplier_normalization_summary, original_columns):
    required_fields = ["supplier_name", "annual_spend", "category"]
    performance_fields = ["on_time_delivery", "prior_year_otd", "defect_rate", "prior_year_defect_rate"]
    contract_fields = ["contract_status"]
    context_fields = ["supplier_criticality", "business_unit", "region", "buyer", "invoice_date"]

    dimensions = []

    def completeness_score(column):
        if column not in results.columns:
            return 0
        return results[column].notna().mean() * 100

    supplier_score = completeness_score("supplier_name")
    spend_score = 100 if "annual_spend" in results.columns and results["annual_spend"].sum() > 0 else 0
    category_score = completeness_score("category")

    performance_available = [col for col in performance_fields if col in results.columns]
    performance_score = (
        sum(completeness_score(col) for col in performance_available) / len(performance_available)
        if performance_available
        else 0
    )

    contract_score = completeness_score("contract_status") if "contract_status" in results.columns else 0

    date_score = completeness_score("invoice_date") if "invoice_date" in results.columns else 0

    duplicate_family_count = 0
    if supplier_normalization_summary is not None and not supplier_normalization_summary.empty:
        duplicate_family_count = int((supplier_normalization_summary["variant_count"] > 1).sum())

    duplicate_score = 100
    if duplicate_family_count > 0:
        duplicate_score = max(70, 100 - duplicate_family_count * 3)

    dimensions.append(("Supplier name completeness", supplier_score))
    dimensions.append(("Spend field usability", spend_score))
    dimensions.append(("Category completeness", category_score))
    dimensions.append(("Performance metric completeness", performance_score))
    dimensions.append(("Contract status completeness", contract_score))
    dimensions.append(("Date validity", date_score))
    dimensions.append(("Duplicate supplier risk", duplicate_score))

    overall_score = round(sum(score for _, score in dimensions) / len(dimensions), 0)

    missing_columns = [
        col for col in required_fields + performance_fields + contract_fields + context_fields
        if col not in results.columns
    ]

    limitations = []

    if "contract_status" not in results.columns:
        limitations.append("Contract coverage risk is limited because contract_status is missing.")
    if not performance_available:
        limitations.append("Supplier performance deterioration analysis is limited because OTD/defect fields are missing.")
    if "supplier_criticality" not in results.columns:
        limitations.append("Critical supplier exposure is limited because supplier_criticality is missing.")
    if duplicate_family_count > 0:
        limitations.append(f"{duplicate_family_count} potential duplicate supplier families require business review.")

    if not limitations:
        limitations.append("No major data limitations detected for the current diagnostic scope.")

    cleanup_actions = [
        "Validate supplier family normalization before consolidation decisions.",
        "Confirm missing or unknown contract status with sourcing and legal teams.",
        "Review supplier criticality values with category owners.",
        "Confirm performance deterioration with operational stakeholders before supplier escalation.",
    ]

    return {
        "overall_score": overall_score,
        "dimensions": dimensions,
        "missing_columns": missing_columns,
        "limitations": limitations,
        "cleanup_actions": cleanup_actions,
        "duplicate_family_count": duplicate_family_count,
    }


# ============================================================
# CHARTS
# ============================================================

def chart_spend_by_category(results):
    if results.empty:
        return None

    data = (
        results.groupby("category", dropna=False)["annual_spend"]
        .sum()
        .reset_index()
        .sort_values("annual_spend", ascending=True)
        .tail(10)
    )

    fig = px.bar(
        data,
        x="annual_spend",
        y="category",
        orientation="h",
        title="Top Categories by Spend",
        labels={"annual_spend": "Spend", "category": "Category"},
    )
    return apply_chart_layout(fig)


def chart_top_suppliers(results):
    supplier_col = "normalized_supplier_name" if "normalized_supplier_name" in results.columns else "supplier_name"

    data = (
        results.groupby(supplier_col, dropna=False)["annual_spend"]
        .sum()
        .reset_index()
        .sort_values("annual_spend", ascending=True)
        .tail(10)
    )

    fig = px.bar(
        data,
        x="annual_spend",
        y=supplier_col,
        orientation="h",
        title="Top Supplier Families by Spend",
        labels={"annual_spend": "Spend", supplier_col: "Supplier"},
    )
    return apply_chart_layout(fig)


def chart_monthly_spend(results):
    if "invoice_date" not in results.columns:
        return None

    data = results.dropna(subset=["invoice_date"]).copy()

    if data.empty:
        return None

    data["month"] = data["invoice_date"].dt.to_period("M").astype(str)

    trend = (
        data.groupby("month")["annual_spend"]
        .sum()
        .reset_index()
        .sort_values("month")
    )

    fig = px.line(
        trend,
        x="month",
        y="annual_spend",
        markers=True,
        title="Monthly Spend Trend",
        labels={"month": "Month", "annual_spend": "Spend"},
    )
    return apply_chart_layout(fig)


def chart_archetype_distribution(supplier_diagnostics):
    if supplier_diagnostics.empty or "action_archetype" not in supplier_diagnostics.columns:
        return None

    data = (
        supplier_diagnostics.groupby("action_archetype")
        .agg(spend=("annual_spend", "sum"), supplier_count=("supplier", "count"))
        .reset_index()
        .sort_values("spend", ascending=True)
    )

    fig = px.bar(
        data,
        x="spend",
        y="action_archetype",
        orientation="h",
        title="Supplier Spend by Action Archetype",
        labels={"spend": "Spend", "action_archetype": "Action Archetype"},
        hover_data=["supplier_count"],
    )
    return apply_chart_layout(fig)


def chart_contract_risk(supplier_diagnostics):
    if supplier_diagnostics.empty or "contract_risk" not in supplier_diagnostics.columns:
        return None

    data = (
        supplier_diagnostics.groupby("contract_risk")["annual_spend"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        data,
        x="contract_risk",
        y="annual_spend",
        title="Spend by Contract Coverage Risk",
        labels={"contract_risk": "Contract Risk", "annual_spend": "Spend"},
    )
    return apply_chart_layout(fig)


def chart_spend_at_risk_top_suppliers(supplier_diagnostics):
    if supplier_diagnostics.empty:
        return None

    data = supplier_diagnostics[supplier_diagnostics["is_spend_at_risk"]].copy()

    if data.empty:
        return None

    data = data.sort_values("annual_spend", ascending=True).tail(10)

    fig = px.bar(
        data,
        x="annual_spend",
        y="supplier",
        color="action_archetype",
        orientation="h",
        title="Top Suppliers Contributing to Spend at Risk",
        labels={"annual_spend": "Spend at Risk", "supplier": "Supplier", "action_archetype": "Archetype"},
    )
    return apply_chart_layout(fig)


# ============================================================
# RENDER TABS
# ============================================================

def render_executive_diagnostic(results, supplier_diagnostics, spend_risk, pipeline, data_readiness):
    st.subheader("Executive Diagnostic")

    total_spend = spend_risk.get("total_spend", 0)
    spend_at_risk = spend_risk.get("spend_at_risk", 0)
    spend_at_risk_pct = spend_risk.get("spend_at_risk_pct", 0)

    contract_gap_spend = spend_risk.get("contract_gap_spend", None)
    contract_gap_display = (
        format_currency(contract_gap_spend)
        if contract_gap_spend is not None
        else "Not evaluated"
    )

    high_priority_actions = int((pipeline["priority"] == "High").sum()) if not pipeline.empty else 0
    supplier_count = supplier_diagnostics["supplier"].nunique() if not supplier_diagnostics.empty else 0
    duplicate_count = data_readiness.get("duplicate_family_count", 0)

    metric_cols = st.columns(6)
    metric_cols[0].metric("Total Spend", format_currency(total_spend))
    metric_cols[1].metric("Suppliers", format_number(supplier_count))
    metric_cols[2].metric("Spend at Risk", format_currency(spend_at_risk))
    metric_cols[3].metric("% at Risk", format_percent(spend_at_risk_pct))
    metric_cols[4].metric("Contract Gap", contract_gap_display)
    metric_cols[5].metric("High Priority Actions", format_number(high_priority_actions))

    st.markdown("### Diagnostic View")

    st.markdown(
        f"""
The tool analyzed **{format_currency(total_spend)}** across **{format_number(supplier_count)} supplier families**.

It identified **{format_currency(spend_at_risk)}** of spend requiring procurement action, equal to **{format_percent(spend_at_risk_pct)}** of analyzed spend.

The current action pipeline contains **{len(pipeline)} recommended management actions**, including **{high_priority_actions} high-priority items**.
"""
    )

    st.markdown("### Contract Coverage")

    if contract_gap_spend is not None:
        st.markdown(
            f"""
Contract gap exposure is estimated at **{contract_gap_display}**.

This reflects suppliers with Medium or High contract coverage risk based on contract status data provided in the uploaded file.
"""
        )
    else:
        st.warning(
            "Contract gap exposure was not evaluated because `contract_status` was not included in the uploaded file. "
            "This is treated as a data limitation, not an assumed supplier risk."
        )

    st.markdown("### Supplier Normalization")

    st.markdown(
        f"""
The diagnostic identified **{duplicate_count} potential duplicate supplier-family patterns** using alias rules and fuzzy matching.

These matches are used to improve spend concentration and action pipeline logic, but should be reviewed before final supplier consolidation decisions.
"""
    )

    if not pipeline.empty:
        st.markdown("### Immediate Leadership Focus")

        top_actions = pipeline.head(5)

        for idx, row in top_actions.iterrows():
            with st.expander(f"{idx + 1}. {row.get('opportunity_action')}"):
                st.markdown(f"**Value / Exposure:** {row.get('value_exposure_display')}")
                st.markdown(f"**Priority:** {row.get('priority')}")
                st.markdown(f"**Recommended Owner:** {row.get('recommended_owner')}")
                st.markdown(f"**Next Step:** {row.get('next_step')}")
                st.markdown(f"**Why Flagged:** {row.get('why_flagged')}")


def render_spend_analytics(results):
    st.subheader("Spend Analytics: Evidence Base")

    st.markdown(
        """
        This section provides the spend concentration and category-level evidence used to support the
        diagnostic findings, spend-at-risk calculations, and action pipeline.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        show_chart_or_message(chart_spend_by_category(results), "Spend by category is not available.", "spend_by_category")

    with col2:
        show_chart_or_message(chart_top_suppliers(results), "Supplier spend chart is not available.", "top_suppliers")

    col3, col4 = st.columns(2)

    with col3:
        show_chart_or_message(chart_monthly_spend(results), "Monthly trend requires a valid invoice or purchase date.", "monthly_trend")

    with col4:
        category_supplier = (
            results.groupby("category")
            .agg(
                spend=("annual_spend", "sum"),
                supplier_count=("normalized_supplier_name", "nunique")
                if "normalized_supplier_name" in results.columns
                else ("supplier_name", "nunique"),
            )
            .reset_index()
            .sort_values("spend", ascending=False)
            .head(15)
        )

        if category_supplier.empty:
            st.info("Category supplier summary is not available.")
        else:
            st.markdown("### Category Concentration Summary")
            display = category_supplier.copy()
            display["spend"] = display["spend"].apply(format_currency)
            st.dataframe(make_display_table(display), use_container_width=True, hide_index=True)


def render_supplier_risk_performance(supplier_diagnostics, spend_risk):
    st.subheader("Supplier Risk & Performance")

    col1, col2 = st.columns(2)

    with col1:
        show_chart_or_message(chart_archetype_distribution(supplier_diagnostics), "Action archetype distribution is not available.", "archetype_distribution")

    with col2:
        show_chart_or_message(chart_contract_risk(supplier_diagnostics), "Contract risk chart requires contract status.", "contract_risk")

    show_chart_or_message(chart_spend_at_risk_top_suppliers(supplier_diagnostics), "No spend-at-risk suppliers were detected.", "top_spend_at_risk")

    st.markdown("### Supplier Diagnostic Table")

    display_columns = [
        "supplier",
        "category",
        "annual_spend",
        "supplier_criticality",
        "contract_status",
        "contract_risk",
        "on_time_delivery",
        "prior_year_otd",
        "defect_rate",
        "prior_year_defect_rate",
        "action_archetype",
        "priority",
        "recommended_action",
        "why_flagged",
    ]

    available = [col for col in display_columns if col in supplier_diagnostics.columns]
    display = supplier_diagnostics[available].copy()

    if "annual_spend" in display.columns:
        display["annual_spend"] = display["annual_spend"].apply(format_currency)

    st.dataframe(make_display_table(display), use_container_width=True, hide_index=True)


def render_action_pipeline(pipeline):
    st.subheader("Procurement Action Pipeline")

    st.markdown(
        """
        This pipeline converts supplier performance, spend, contract, and sourcing signals into management actions.
        It is designed to support consulting-style prioritization and should be validated with category owners before execution.
        """
    )

    if pipeline.empty:
        st.warning("No action pipeline items were generated.")
        return

    cols = st.columns(4)
    cols[0].metric("Pipeline Actions", format_number(len(pipeline)))
    cols[1].metric("High Priority", format_number((pipeline["priority"] == "High").sum()))
    cols[2].metric("Medium Priority", format_number((pipeline["priority"] == "Medium").sum()))
    cols[3].metric("Value / Exposure", format_currency(pipeline["value_exposure"].sum()))

    display_columns = [
        "opportunity_action",
        "supplier_or_category",
        "category",
        "action_archetype",
        "value_exposure_display",
        "priority",
        "confidence",
        "complexity",
        "recommended_owner",
        "next_step",
        "validation_required",
        "status",
        "why_flagged",
    ]

    display = pipeline[display_columns].copy()
    st.dataframe(make_display_table(display), use_container_width=True, hide_index=True)

    st.markdown("### High-Priority Action Cards")

    high_priority = pipeline[pipeline["priority"] == "High"].copy()

    if high_priority.empty:
        st.success("No high-priority action items were generated.")
    else:
        for _, row in high_priority.head(5).iterrows():
            st.markdown(
                f"""
                <div class="section-card">
                    <h4>{row.get("opportunity_action")}</h4>
                    <p><strong>Value / Exposure:</strong> {row.get("value_exposure_display")}</p>
                    <p><strong>Recommended Owner:</strong> {row.get("recommended_owner")}</p>
                    <p><strong>Next Step:</strong> {row.get("next_step")}</p>
                    <p><strong>Validation Required:</strong> {row.get("validation_required")}</p>
                    <p><strong>Why Flagged:</strong> {row.get("why_flagged")}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    csv_export = pipeline.drop(columns=["value_exposure"], errors="ignore").to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Action Pipeline CSV",
        data=csv_export,
        file_name="procurement_action_pipeline.csv",
        mime="text/csv",
    )


def render_qbr_briefing(supplier_diagnostics, results):
    st.subheader("Supplier QBR Briefing Mode")

    if supplier_diagnostics.empty:
        st.warning("No supplier diagnostics available.")
        return

    supplier_list = sorted(supplier_diagnostics["supplier"].dropna().astype(str).unique())

    selected_supplier = st.selectbox(
        "Select supplier for QBR briefing",
        supplier_list,
    )

    row = supplier_diagnostics[supplier_diagnostics["supplier"] == selected_supplier].iloc[0]

    supplier_col = "normalized_supplier_name" if "normalized_supplier_name" in results.columns else "supplier_name"
    supplier_rows = results[results[supplier_col] == selected_supplier]

    variants = []
    if "original_supplier_name" in supplier_rows.columns:
        variants = sorted(supplier_rows["original_supplier_name"].dropna().astype(str).unique())

    flags = get_performance_flags(row)

    spend = row.get("annual_spend", 0)
    category = row.get("category", "Unclassified")
    criticality = row.get("supplier_criticality", "Not available")
    contract_status = row.get("contract_status", "Not available")
    contract_risk = row.get("contract_risk", "Not evaluated")
    action_archetype = row.get("action_archetype", "Maintain / Monitor")
    recommended_action = row.get("recommended_action", "Continue monitoring through standard supplier management process.")
    why_flagged = row.get("why_flagged", "No diagnostic flag available.")

    st.markdown(f"### Supplier QBR Briefing — {selected_supplier}")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Analyzed Spend", format_currency(spend))
    metric_cols[1].metric("Category", str(category))
    metric_cols[2].metric("Criticality", str(criticality))
    metric_cols[3].metric("Contract Risk", str(contract_risk))

    with st.container(border=True):
        st.markdown("#### 1. Supplier Snapshot")
        st.markdown(
            f"**{selected_supplier}** represents **{format_currency(spend)}** in analyzed spend "
            f"across the **{category}** category. Supplier criticality is **{criticality}**."
        )

    # Only show performance section if at least one performance field exists.
    performance_fields = [
        "on_time_delivery",
        "prior_year_otd",
        "defect_rate",
        "prior_year_defect_rate",
        "lead_time_days",
    ]

    has_performance_data = any(
        field in supplier_diagnostics.columns and pd.notna(row.get(field))
        for field in performance_fields
    )

    if has_performance_data:
        with st.container(border=True):
            st.markdown("#### 2. Performance Summary")

            perf_cols = st.columns(5)

            if "on_time_delivery" in supplier_diagnostics.columns:
                perf_cols[0].metric("On-Time Delivery", format_percent(row.get("on_time_delivery")))

            if "prior_year_otd" in supplier_diagnostics.columns:
                perf_cols[1].metric("Prior-Year OTD", format_percent(row.get("prior_year_otd")))

            if "defect_rate" in supplier_diagnostics.columns:
                perf_cols[2].metric("Defect Rate", format_percent(row.get("defect_rate")))

            if "prior_year_defect_rate" in supplier_diagnostics.columns:
                perf_cols[3].metric("Prior-Year Defect", format_percent(row.get("prior_year_defect_rate")))

            if "lead_time_days" in supplier_diagnostics.columns:
                lead_time = row.get("lead_time_days")
                lead_time_display = "Not available" if pd.isna(lead_time) else f"{float(lead_time):.1f} days"
                perf_cols[4].metric("Lead Time", lead_time_display)
    else:
        with st.container(border=True):
            st.markdown("#### 2. Performance Summary")
            st.info(
                "Supplier performance metrics such as OTD, defect rate, and lead time were not available "
                "in this ERP spend file. The QBR view is therefore based on spend, category, contract, "
                "supplier criticality, and diagnostic flags."
            )

    with st.container(border=True):
        st.markdown("#### 3. Contract Position")
        st.markdown(
            f"- **Contract status:** {contract_status}\n"
            f"- **Contract risk:** {contract_risk}"
        )

    with st.container(border=True):
        st.markdown("#### 4. Supplier Action Archetype")
        st.markdown(f"**{action_archetype}**")
        st.markdown(
            "This archetype translates the supplier’s spend, contract, criticality, and available performance "
            "signals into a procurement management action."
        )

    with st.container(border=True):
        st.markdown("#### 5. Why Supplier Was Flagged")
        st.markdown(str(why_flagged))

    with st.container(border=True):
        st.markdown("#### 6. Recommended Action")
        st.markdown(str(recommended_action))

    with st.container(border=True):
        st.markdown("#### 7. Internal Decision Required")
        st.markdown(
            "Procurement leadership should decide whether to **maintain the current relationship**, "
            "**review contract coverage**, **shift volume**, **rationalize the supplier**, or "
            "**escalate the supplier for further review** based on validated business impact."
        )

    if variants and len(variants) > 1:
        with st.expander("Supplier Family Variants Detected"):
            variant_df = pd.DataFrame(
                {
                    "original_supplier_variant": variants,
                    "normalized_supplier": selected_supplier,
                }
            )
            st.dataframe(make_display_table(variant_df), use_container_width=True, hide_index=True)

    st.markdown("### Questions to Ask Supplier / Category Owner")

    if has_performance_data:
        questions = [
            "What caused the delivery, quality, or lead-time performance change?",
            "Which regions, sites, lanes, business units, or services are driving the issue?",
            "What corrective action plan will restore performance within 60 days?",
            "Are SLA credits, remedies, or service recovery commitments available?",
            "Is the supplier’s current pricing justified given recent performance?",
            "Should volume be maintained, reduced, shifted, or competitively reviewed?",
        ]
    else:
        questions = [
            "Is this supplier correctly mapped to the normalized supplier family?",
            "Is the category assignment accurate and useful for sourcing analysis?",
            "Is the supplier currently under contract, and is the contract metadata complete?",
            "Is the spend addressable, or is it constrained by business, technical, or location requirements?",
            "Should this supplier be part of a consolidation, contract review, or preferred-supplier strategy?",
            "What additional data is needed before making a sourcing decision?",
        ]

    for question in questions:
        st.markdown(f"- {question}")

    if flags:
        st.markdown("### Performance Deterioration Flags")
        st.dataframe(make_display_table(pd.DataFrame(flags)), use_container_width=True, hide_index=True)


def render_cpo_briefing(supplier_diagnostics, pipeline, spend_risk, data_readiness):
    st.subheader("CPO Executive Briefing")

    total_spend = spend_risk.get("total_spend", 0)
    spend_at_risk = spend_risk.get("spend_at_risk", 0)
    spend_at_risk_pct = spend_risk.get("spend_at_risk_pct", 0)

    contract_gap = spend_risk.get("contract_gap_spend", None)
    contract_gap_display = (
        format_currency(contract_gap)
        if contract_gap is not None
        else "Not evaluated"
    )

    total_actions = len(pipeline) if pipeline is not None and not pipeline.empty else 0
    high_priority_count = int((pipeline["priority"] == "High").sum()) if total_actions > 0 else 0
    medium_priority_count = int((pipeline["priority"] == "Medium").sum()) if total_actions > 0 else 0

    supplier_count = (
        supplier_diagnostics["supplier"].nunique()
        if supplier_diagnostics is not None and not supplier_diagnostics.empty
        else 0
    )

    # Separate supplier-level risk from pipeline value.
    # This prevents weird CPO language like "$0 at risk" while the tool still has action items.
    if spend_at_risk > 0:
        risk_summary = (
            f"The diagnostic identified **{format_currency(spend_at_risk)}**, or "
            f"**{format_percent(spend_at_risk_pct)}** of analyzed spend, associated with suppliers requiring action "
            "based on available risk, performance, criticality, or contract signals."
        )
    elif total_actions > 0:
        risk_summary = (
            "The uploaded file did **not provide enough supplier-level risk evidence** to quantify supplier-level "
            "spend-at-risk. However, the diagnostic still generated an action pipeline based on available sourcing, "
            "spend concentration, supplier fragmentation, and validation signals."
        )
    else:
        risk_summary = (
            "The uploaded file did not provide enough evidence to quantify supplier-level spend-at-risk or generate "
            "a meaningful action pipeline. Additional supplier performance, contract, and criticality fields would "
            "improve the diagnostic."
        )

    # Top attention areas
    top_attention = []

    if pipeline is not None and not pipeline.empty:
        top_attention = pipeline.head(5)["opportunity_action"].astype(str).tolist()

    # Contract wording should not overreach when contract_status is missing.
    if contract_gap is not None:
        contract_summary = (
            f"Contract coverage exposure is estimated at **{contract_gap_display}** based on contract status data "
            "provided in the file. Procurement should validate whether this represents true unmanaged spend, expired "
            "coverage, or incomplete metadata."
        )
    else:
        contract_summary = (
            "Contract coverage was **not evaluated** because `contract_status` was not included in the uploaded file. "
            "This should be treated as a data limitation, not an assumed contract risk."
        )

    # -----------------------------
    # Header KPI cards
    # -----------------------------
    metric_cols = st.columns(5)
    metric_cols[0].metric("Total Spend", format_currency(total_spend))
    metric_cols[1].metric("Supplier Families", format_number(supplier_count))
    metric_cols[2].metric("Pipeline Actions", format_number(total_actions))
    metric_cols[3].metric("High Priority", format_number(high_priority_count))
    metric_cols[4].metric("Data Readiness", f"{data_readiness.get('overall_score', 0)} / 100")

    # -----------------------------
    # Card 1: Executive Summary
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 1. Executive Summary")

        st.markdown(
            f"""
The diagnostic reviewed **{format_currency(total_spend)}** in supplier spend across **{format_number(supplier_count)} supplier families**.

{risk_summary}

The current action pipeline includes **{total_actions} recommended actions**, including **{high_priority_count} high-priority** and **{medium_priority_count} medium-priority** items.

This output should be used as a **procurement triage view**: it identifies where leadership attention, category-owner validation, and follow-up analysis are required before action.
"""
        )

    # -----------------------------
    # Card 2: Top Attention Areas
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 2. Top Attention Areas")

        if top_attention:
            for item in top_attention:
                st.markdown(f"- {item}")
        else:
            st.info("No top attention areas were generated from the available data.")

    # -----------------------------
    # Card 3: Contract Coverage
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 3. Contract Coverage")

        if contract_gap is not None:
            st.markdown(contract_summary)
        else:
            st.warning(contract_summary)

    # -----------------------------
    # Card 4: Leadership Decisions Needed
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 4. Decisions Needed from Procurement Leadership")

        st.markdown(
            """
- Which high-priority pipeline items should be validated first?
- Which category owners should own each opportunity?
- Which findings require supplier QBR escalation?
- Which sourcing opportunities should move into business-case validation?
- What additional data is needed before making contract, consolidation, or supplier-risk decisions?
"""
        )

    # -----------------------------
    # Card 5: 30 / 60 / 90 Day Action Plan
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 5. 30 / 60 / 90 Day Action Plan")

        plan = pd.DataFrame(
            [
                {
                    "timeframe": "30 days",
                    "actions": "Validate the highest-priority action pipeline items, confirm owners, and review data gaps with category stakeholders.",
                },
                {
                    "timeframe": "60 days",
                    "actions": "Convert validated items into corrective action plans, sourcing reviews, supplier consolidation reviews, or contract validation workstreams.",
                },
                {
                    "timeframe": "90 days",
                    "actions": "Track approved actions through execution, monitor realized savings or risk mitigation, and establish recurring supplier performance governance.",
                },
            ]
        )

        st.dataframe(make_display_table(plan), use_container_width=True, hide_index=True)

    # -----------------------------
    # Card 6: Methodology / Caution
    # -----------------------------
    with st.container(border=True):
        st.markdown("### 6. Methodology and Caution")

        st.markdown(
            f"""
Current data readiness score is **{data_readiness.get("overall_score", 0)} / 100**.

The briefing is generated from rule-based diagnostics using the fields available in the uploaded file. The tool avoids assuming supplier risk when key fields are missing. For example, missing contract status is treated as a **data limitation**, not as evidence of contract risk.

Recommendations should be validated with category owners before supplier escalation, consolidation, contract review, or sourcing decisions.
"""
        )


def render_data_readiness(results, supplier_normalization_summary, data_readiness, mapping):
    st.subheader("Data Readiness / Methodology")

    cols = st.columns(3)
    cols[0].metric("Data Readiness Score", f"{data_readiness.get('overall_score', 0)} / 100")
    cols[1].metric("Rows Analyzed", format_number(len(results)))
    cols[2].metric("Potential Duplicate Families", format_number(data_readiness.get("duplicate_family_count", 0)))

    st.markdown("### Data Readiness Dimensions")

    dimension_df = pd.DataFrame(
        data_readiness["dimensions"],
        columns=["dimension", "score"],
    )

    dimension_df["score"] = dimension_df["score"].round(1)
    st.dataframe(make_display_table(dimension_df), use_container_width=True, hide_index=True)

    st.markdown("### Data Limitations")
    for item in data_readiness["limitations"]:
        st.markdown(f"- {item}")

    st.markdown("### Recommended Cleanup Actions")
    for item in data_readiness["cleanup_actions"]:
        st.markdown(f"- {item}")

    st.markdown("### Methodology Notes")
    st.markdown(
        """
        - Supplier action archetypes are assigned using rule-based logic based on spend, criticality, contract coverage, and performance deterioration.
        - Spend-at-risk includes suppliers flagged as Executive Escalation, Watchlist Supplier, Contract Review Priority, Rationalization Candidate, or Alternate Source Required.
        - Fuzzy supplier family normalization uses alias rules and RapidFuzz similarity matching.
        - Savings estimates are directional and intended for opportunity prioritization, not finance-approved savings.
        - All recommendations require procurement validation before business action.
        """
    )

    if mapping:
        with st.expander("Review column mapping used by the tool"):
            mapping_df = pd.DataFrame(
                [{"uploaded_column": key, "mapped_to": value} for key, value in mapping.items()]
            )
            st.dataframe(make_display_table(mapping_df), use_container_width=True, hide_index=True)

    if supplier_normalization_summary is not None and not supplier_normalization_summary.empty:
        with st.expander("Review supplier family normalization"):
            display = supplier_normalization_summary.copy()
            display["total_spend"] = display["total_spend"].apply(format_currency)
            st.dataframe(make_display_table(display), use_container_width=True, hide_index=True)


# ============================================================
# MAIN APP
# ============================================================

def main():
    apply_global_styles()

    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-label">Sid Shetty Portfolio Project</div>
            <div class="hero-title">Supplier Performance Diagnostic Workbench</div>
            <div class="hero-subtitle">
                A consulting-style supplier performance and spend diagnostic workbench that converts messy supplier data
                into supplier risk segmentation, spend-at-risk exposure, contract coverage gaps, QBR talking points,
                and executive-ready procurement recommendations.
                <br><br>
                This tool is designed to mirror the early diagnostic phase of a procurement performance review. It does
                not just summarize spend; it identifies which suppliers require action, why they were flagged, what
                business exposure is attached, and what procurement leadership should validate next.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.header("Data Input")

    input_mode = st.sidebar.radio(
        "Choose input mode",
        ["Upload file", "Use demo data"],
    )

    raw_df = None
    data_source_label = None

    if input_mode == "Upload file":
        uploaded_file = st.sidebar.file_uploader(
            "Upload supplier spend or performance file",
            type=["csv", "xlsx", "xls"],
        )

        if uploaded_file is not None:
            try:
                raw_df, data_source_label = load_uploaded_data(uploaded_file)
            except Exception as error:
                st.error(f"Unable to read uploaded file: {error}")
                return
        else:
            st.info("Upload a CSV or Excel file, or switch to demo data in the sidebar.")
            return

    else:
        demo_type = st.sidebar.selectbox(
            "Demo dataset",
            [
                "Mixed indirect spend supplier portfolio",
                "Logistics supplier performance",
                "IT services supplier performance",
                "MRO supplier performance",
                "Professional services supplier performance",
            ],
        )

        raw_df = build_demo_dataset(demo_type)
        data_source_label = demo_type

    supplier_match_threshold = st.sidebar.slider(
        "Supplier fuzzy match threshold",
        min_value=80,
        max_value=100,
        value=90,
        step=1,
        help="Higher threshold = stricter matching. Lower threshold = more aggressive grouping.",
    )

    try:
        results, column_mapping = standardize_columns(raw_df)
    except Exception as error:
        st.error(f"Unable to process the dataset: {error}")
        return

    results, supplier_normalization_summary = apply_supplier_normalization(
        results,
        threshold=supplier_match_threshold,
    )

    supplier_summary = calculate_supplier_summary(results)
    supplier_diagnostics = assign_supplier_archetypes(supplier_summary)
    spend_risk = calculate_spend_at_risk(supplier_diagnostics)
    opportunities = create_sourcing_opportunities(results, supplier_diagnostics)
    action_pipeline = create_action_pipeline(results, supplier_diagnostics, opportunities)
    data_readiness = calculate_data_readiness(
        results,
        supplier_normalization_summary,
        raw_df.columns.tolist(),
    )

    st.sidebar.success(f"Data loaded: {data_source_label}")
    st.sidebar.metric("Rows analyzed", format_number(len(results)))
    st.sidebar.metric("Spend analyzed", format_currency(results["annual_spend"].sum()))

    tabs = st.tabs(
        [
            "Executive Diagnostic",
            "Spend Analytics",
            "Supplier Risk & Performance",
            "Action Pipeline",
            "QBR Briefing",
            "CPO Briefing",
            "Data Readiness",
        ]
    )

    with tabs[0]:
        render_executive_diagnostic(
            results,
            supplier_diagnostics,
            spend_risk,
            action_pipeline,
            data_readiness,
        )

    with tabs[1]:
        render_spend_analytics(results)

    with tabs[2]:
        render_supplier_risk_performance(supplier_diagnostics, spend_risk)

    with tabs[3]:
        render_action_pipeline(action_pipeline)

    with tabs[4]:
        render_qbr_briefing(supplier_diagnostics, results)

    with tabs[5]:
        render_cpo_briefing(
            supplier_diagnostics,
            action_pipeline,
            spend_risk,
            data_readiness,
        )

    with tabs[6]:
        render_data_readiness(
            results,
            supplier_normalization_summary,
            data_readiness,
            column_mapping,
        )


if __name__ == "__main__":
    main()