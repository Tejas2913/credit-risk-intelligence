"""
Module 1: EDA / Analytics Route Handlers
=========================================
Exposes dataset summary and the 6 core business insights established in the notebook EDA.
Queries `data/credit_risk_analytics.db` directly in read-only mode.
"""

import logging
import os
import sqlite3
from typing import List

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    EDAInsight,
    EDAInsightsResponse,
    EDASummaryResponse,
    InsightChartData,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/eda", tags=["EDA & Business Insights"])

DEFAULT_ANALYTICS_DB = os.path.join("data", "credit_risk_analytics.db")

# Authoritative Business Insights dataset from EDA notebook
AUTHORITATIVE_INSIGHTS: List[dict] = [
    {
        "id": 1,
        "title": "Distribution of Loan Default Outcomes",
        "metric": "8.07% Overall Portfolio Default Rate (Class Imbalance: 11.4:1)",
        "chart": {
            "labels": ["Non-Default", "Default"],
            "values": [282686.0, 24825.0],
            "counts": [282686, 24825]
        },
        "interpretation": (
            "The dataset is highly imbalanced, with 282,686 applicants (91.93%) classified as non-default "
            "and 24,825 applicants (8.07%) classified as default. This shows that default cases represent "
            "a relatively small proportion of the overall applicant population. Therefore, a credit-risk "
            "prediction model should not rely on accuracy alone and must explicitly address class imbalance "
            "to identify high-risk applicants effectively."
        )
    },
    {
        "id": 2,
        "title": "Default Rate by Previous Application Outcome",
        "metric": "12.0% Default Rate for Previously Refused vs 7.6% for Previously Approved",
        "chart": {
            "labels": ["Refused", "Canceled", "Unused offer", "Approved"],
            "values": [12.0, 9.2, 8.2, 7.6],
            "counts": [245390, 259441, 22771, 886099]
        },
        "interpretation": (
            "Applicants whose previous loan applications were refused have the highest observed default rate "
            "at 12.0%, compared with 7.6% among applicants whose previous applications were approved. This indicates "
            "that previous application outcomes contain useful information about future credit risk. Historical "
            "refusal behavior can therefore be incorporated into features such as previous refusal count and refusal "
            "rate when building the credit-risk model."
        )
    },
    {
        "id": 3,
        "title": "Default Rate by Historical POS/CASH Delinquency",
        "metric": "12.0% Default Rate for High DPD vs 7.7% for No Recorded DPD",
        "chart": {
            "labels": ["No DPD", "Low DPD", "Moderate DPD", "High DPD"],
            "values": [7.7, 8.5, 11.0, 12.0],
            "counts": [234761, 23946, 22961, 7776]
        },
        "interpretation": (
            "The observed default rate increases with the severity of historical POS/CASH delinquency. Applicants with "
            "no recorded DPD have a default rate of 7.7%, while applicants in the high-DPD group have a default rate of "
            "12.0%. This indicates that repeated or more severe payment delinquency in previous POS/CASH accounts is "
            "associated with increased credit risk. Features such as DPD frequency, maximum DPD, and delinquency ratio "
            "can therefore provide valuable signals for risk prediction."
        )
    },
    {
        "id": 4,
        "title": "Default Rate by Credit Card Utilization",
        "metric": "19.5% Default Rate for Very High Utilization vs 5.4% for Zero Utilization",
        "chart": {
            "labels": ["No utilization", "Low", "Moderate", "High", "Very High"],
            "values": [5.4, 6.4, 9.1, 12.6, 19.5],
            "counts": [26604, 19857, 18558, 15569, 5448]
        },
        "interpretation": (
            "Credit-card utilization shows a strong relationship with observed default risk. The default rate increases "
            "from 5.4% for applicants with no utilization to 19.5% for applicants with very high utilization. This represents "
            "a substantial difference in observed risk and suggests that applicants using a large proportion of their "
            "available credit may have greater difficulty managing additional credit obligations. Therefore, average and "
            "maximum credit-card utilization are important behavioral features for the credit-risk model."
        )
    },
    {
        "id": 5,
        "title": "Default Rate by Historical Installment Payment Behavior",
        "metric": "13.5% Default Rate for Low Payment Ratio vs 7.5% for Full Payment",
        "chart": {
            "labels": ["Low payment", "Near full payment", "Full payment", "Overpayment"],
            "values": [13.5, 10.4, 7.5, 6.2],
            "counts": [15122, 49284, 193915, 33319]
        },
        "interpretation": (
            "Historical installment payment behavior shows a clear relationship with default risk. Applicants in the "
            "low-payment group have the highest observed default rate at 13.5%, while applicants making full payments have "
            "a lower default rate of 7.5%. This suggests that consistently paying a larger proportion of scheduled "
            "installment amounts is associated with lower credit risk. Features such as overall payment ratio and "
            "underpayment ratio can therefore help the model assess repayment reliability."
        )
    },
    {
        "id": 6,
        "title": "Default Rate by Historical Late-Payment Frequency",
        "metric": "12.0% Default Rate for High Late Payments vs 6.8% for Zero Late Payments",
        "chart": {
            "labels": ["No late payments", "Low", "Moderate", "High"],
            "values": [6.8, 7.0, 9.4, 12.0],
            "counts": [136644, 40212, 77325, 37462]
        },
        "interpretation": (
            "The observed default rate generally increases as the frequency of historical late payments increases. "
            "Applicants with no late payments have a default rate of 6.8%, compared with 12.0% among applicants in the "
            "high late-payment group. This indicates that repeated delays in historical installment payments are "
            "associated with increased future credit risk. Consequently, late-payment count and late-payment ratio are "
            "useful behavioral features for the credit-risk prediction model."
        )
    }
]


@router.get("/summary", response_model=EDASummaryResponse, summary="Get dataset summary and baseline risk statistics")
def get_eda_summary() -> EDASummaryResponse:
    """
    Retrieve high-level portfolio overview statistics derived from the analytics database.
    Returns total applicants, default vs non-default counts, default rate, and feature counts.
    """
    if not os.path.exists(DEFAULT_ANALYTICS_DB):
        logger.error("Analytics database not found at configured path.")
        raise HTTPException(
            status_code=500,
            detail="Analytics database is unavailable. Please contact the administrator."
        )

    try:
        conn = sqlite3.connect(f"file:{DEFAULT_ANALYTICS_DB}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*), SUM(target), AVG(target) FROM applicant_analytics;")
        row = cur.fetchone()
        conn.close()

        total = int(row[0]) if row and row[0] is not None else 0
        defaults = int(row[1]) if row and row[1] is not None else 0
        non_defaults = total - defaults
        default_rate = float(row[2]) if row and row[2] is not None else 0.0

        return EDASummaryResponse(
            total_applicants=total,
            default_applicants=defaults,
            non_default_applicants=non_defaults,
            default_rate=round(default_rate, 6),
            default_rate_pct=round(default_rate * 100, 2),
            total_features=231,
            numerical_features=106,
            categorical_features=16
        )
    except Exception as exc:
        logger.error("Error retrieving EDA summary: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="A database error occurred while retrieving the portfolio summary.")


@router.get("/insights", response_model=EDAInsightsResponse, summary="Get 6 core business insights with chart data")
def get_eda_insights() -> EDAInsightsResponse:
    """
    Retrieve the 6 core business insights from the exploratory data analysis.
    Provides chart-ready labels, values, sample counts, and business interpretations.
    """
    insights_list = [
        EDAInsight(
            id=item["id"],
            title=item["title"],
            metric=item["metric"],
            chart=InsightChartData(**item["chart"]),
            interpretation=item["interpretation"]
        )
        for item in AUTHORITATIVE_INSIGHTS
    ]
    return EDAInsightsResponse(insights=insights_list)
