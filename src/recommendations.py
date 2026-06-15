import pandas as pd


def format_currency(value: float) -> str:
    """
    Format a numeric value as US currency.
    """
    return f"${value:,.0f}"


def create_supplier_finding(
    supplier_row: pd.Series,
) -> dict[str, str | float]:
    """
    Create one evidence-backed management finding for a supplier.
    """
    supplier_name = supplier_row["supplier_name"]
    category = supplier_row["category"]
    annual_spend = supplier_row["annual_spend"]
    category_share = supplier_row["category_spend_share_pct"]
    attention_score = supplier_row["supplier_attention_score"]
    attention_level = supplier_row["attention_level"]
    confidence = supplier_row["data_confidence_pct"]

    observations = []
    implications = []
    next_steps = []

    if supplier_row.get(
        "high_spend_exposure_flag",
        False,
    ):
        observations.append(
            f"represents {category_share:.1f}% "
            f"of {category} spend"
        )
        implications.append(
            "the category has meaningful financial exposure "
            "to this supplier"
        )
        next_steps.append(
            "review dependency, contract coverage, and "
            "contingency options"
        )

    if supplier_row.get(
        "delivery_deterioration_flag",
        False,
    ):
        otd_change = supplier_row[
            "otd_change_pct_points"
        ]

        observations.append(
            f"on-time delivery declined by "
            f"{abs(otd_change):.1f} percentage points"
        )
        implications.append(
            "continued delivery deterioration may affect "
            "service continuity or internal operations"
        )
        next_steps.append(
            "validate the cause of delivery deterioration "
            "and agree on a corrective action plan"
        )

    if supplier_row.get(
        "quality_deterioration_flag",
        False,
    ):
        quality_change = supplier_row[
            "defect_rate_change_pct_points"
        ]

        observations.append(
            f"defect rate increased by "
            f"{quality_change:.1f} percentage points"
        )
        implications.append(
            "worsening quality may increase rework, service "
            "issues, or total cost"
        )
        next_steps.append(
            "review defect drivers and confirm corrective "
            "quality actions"
        )

    if supplier_row.get(
        "supplier_criticality",
        ""
    ) == "High":
        observations.append(
            "is classified as highly critical"
        )
        implications.append(
            "performance issues may have greater operational "
            "impact than for a non-critical supplier"
        )
        next_steps.append(
            "confirm business-continuity and escalation plans"
        )

    if supplier_row.get(
        "missing_performance_data_flag",
        False,
    ):
        observations.append(
            "has incomplete delivery or quality data"
        )
        implications.append(
            "the current assessment has reduced confidence"
        )
        next_steps.append(
            "complete missing performance records before "
            "making a final supplier decision"
        )

    if not observations:
        observations.append(
            "does not currently trigger a major review flag"
        )
        implications.append(
            "no immediate management escalation is indicated "
            "by the available data"
        )
        next_steps.append(
            "continue routine performance monitoring"
        )

    observation_text = (
        f"{supplier_name} has annual spend of "
        f"{format_currency(annual_spend)} and "
        + "; ".join(observations)
        + "."
    )

    implication_text = (
        "This matters because "
        + "; ".join(dict.fromkeys(implications))
        + "."
    )

    next_step_text = (
        "; ".join(dict.fromkeys(next_steps))
        + "."
    )

    return {
        "supplier_name": supplier_name,
        "category": category,
        "annual_spend": annual_spend,
        "supplier_attention_score": attention_score,
        "attention_level": attention_level,
        "data_confidence_pct": confidence,
        "observation": observation_text,
        "implication": implication_text,
        "suggested_next_step": next_step_text,
    }


def create_supplier_findings(
    data: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Create findings for the highest-attention suppliers.
    """
    ranked_suppliers = (
        data.sort_values(
            by="supplier_attention_score",
            ascending=False,
        )
        .head(top_n)
    )

    findings = [
        create_supplier_finding(row)
        for _, row in ranked_suppliers.iterrows()
    ]

    return pd.DataFrame(findings)


def create_category_findings(
    category_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create evidence-backed category review findings.
    """
    findings = []

    for _, row in category_metrics.iterrows():
        observations = []
        implications = []
        next_steps = []

        if row["concentration_risk_flag"]:
            observations.append(
                f"the largest supplier represents "
                f"{row['top_supplier_share_pct']:.1f}% "
                f"of category spend"
            )
            implications.append(
                "the category may be exposed to supplier "
                "dependency or continuity risk"
            )
            next_steps.append(
                "review contingency options and supplier "
                "dependency"
            )

        if row["fragmentation_review_flag"]:
            observations.append(
                f"spend is distributed across "
                f"{int(row['supplier_count'])} suppliers, "
                f"while the top three represent only "
                f"{row['top_3_supplier_share_pct']:.1f}%"
            )
            implications.append(
                "the supplier base may be administratively "
                "complex or insufficiently consolidated"
            )
            next_steps.append(
                "validate whether supplier-base rationalization "
                "is operationally appropriate"
            )

        if row["tail_spend_review_flag"]:
            observations.append(
                f"{row['tail_supplier_pct']:.1f}% of suppliers "
                f"fall below the tail-spend threshold and "
                f"represent {row['tail_spend_pct']:.1f}% "
                f"of category spend"
            )
            implications.append(
                "the category may carry disproportionate "
                "administrative effort relative to spend"
            )
            next_steps.append(
                "review low-value suppliers, buying channels, "
                "and transaction efficiency"
            )

        if not observations:
            continue

        findings.append(
            {
                "category": row["category"],
                "total_category_spend": row[
                    "total_category_spend"
                ],
                "observation": (
                    f"{row['category']}: "
                    + "; ".join(observations)
                    + "."
                ),
                "implication": (
                    "This may indicate that "
                    + "; ".join(
                        dict.fromkeys(implications)
                    )
                    + "."
                ),
                "suggested_next_step": (
                    "; ".join(
                        dict.fromkeys(next_steps)
                    )
                    + "."
                ),
            }
        )

    return pd.DataFrame(findings)