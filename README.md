# Supplier Performance & Concentration Briefing Tool

A Streamlit-based analytics application that identifies supplier relationships and spend categories that may deserve management review based on supplier performance, spend exposure, category concentration, and data completeness.

This project is designed as a portfolio-ready procurement analytics tool. It uses synthetic supplier data and a transparent, rules-based scoring methodology.

## What the App Does

The application helps users review supplier and category-level risk signals by analyzing:

* supplier spend exposure
* year-over-year on-time delivery movement
* year-over-year defect-rate movement
* supplier criticality
* category concentration
* category fragmentation
* tail-supplier patterns
* missing data and data confidence

The tool produces a management-attention list, evidence-backed supplier findings, category findings, interactive charts, and downloadable briefing exports.

## Key Features

### Supplier Attention Scoring

The app calculates an illustrative supplier attention score using:

| Component            | Weight |
| -------------------- | -----: |
| Spend exposure       |    30% |
| Delivery risk        |    25% |
| Quality risk         |    25% |
| Supplier criticality |    20% |

The score is not intended to make final sourcing decisions. It is designed to prioritize where further review may be useful.

### Category Concentration Analysis

The app evaluates category-level concentration and fragmentation using:

* top supplier share
* top three supplier share
* top five supplier share
* suppliers required to reach 80% of category spend
* HHI concentration index
* tail-supplier count
* tail-spend percentage

### Evidence-Backed Findings

The app converts metrics into structured management language:

* observation
* implication
* suggested next step

The findings are generated through transparent rules, not black-box AI.

### Interactive Dashboard

The Streamlit interface includes:

* executive overview KPIs
* supplier attention table
* supplier detail view
* category concentration chart
* supplier spend vs. attention scatter plot
* category and attention-level filters
* methodology tab

### Exportable Executive Briefing

The app can export:

* supplier briefing CSV
* category briefing CSV
* multi-tab Excel executive briefing workbook

The Excel workbook includes:

* Supplier Findings
* Category Findings
* Top Supplier Scores
* Category Metrics
* Methodology

## Project Structure

```text
supplier-performance-briefing/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── sample/
│       └── synthetic_supplier_performance.csv
├── src/
│   ├── category_metrics.py
│   ├── charts.py
│   ├── cleaning.py
│   ├── data_generation.py
│   ├── export.py
│   ├── ingestion.py
│   ├── recommendations.py
│   ├── scoring.py
│   ├── supplier_metrics.py
│   └── validation.py
└── tests/
    └── __init__.py
```

## How to Run Locally

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd supplier-performance-briefing
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

On Windows:

```bash
.venv\Scripts\activate
```

On Mac/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the app

```bash
python -m streamlit run app.py
```

The app should open at:

```text
http://localhost:8501
```

## Data

The included dataset is synthetic and generated for demonstration purposes. It is designed to include realistic supplier-management scenarios such as:

* high-spend suppliers with deteriorating delivery performance
* suppliers with worsening quality
* concentrated categories
* fragmented categories
* tail suppliers
* missing performance data
* duplicate-like supplier names

No employer, client, confidential, or proprietary benchmarking data is used.

## Methodology Notes

The supplier attention score is illustrative and rules-based. Python calculates the metrics and findings using transparent logic.

Missing data does not automatically increase supplier risk. Instead, the app calculates supplier scores from available evidence and separately reports data confidence.

The findings should be interpreted as review hypotheses, not final sourcing decisions.

## Limitations

This prototype does not currently:

* perform fuzzy supplier-name matching
* support transaction-level spend aggregation
* benchmark suppliers against external market data
* calculate guaranteed savings
* replace supplier due diligence
* generate final sourcing recommendations

## Future Enhancements

Potential future improvements include:

* fuzzy supplier-name normalization
* transaction-level upload support
* configurable scoring weights
* category-specific thresholds
* expanded Excel formatting
* automated PowerPoint briefing export
* optional LLM-generated executive summary based only on calculated findings
* deployment on Streamlit Community Cloud

## Portfolio Context

This project demonstrates applied analytics, procurement domain knowledge, Python development, data cleaning, transparent scoring logic, interactive dashboarding, and executive-ready export design.
