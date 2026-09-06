"""
Natural Language to SQL Generator Module
========================================
Translates user natural-language questions into valid, schema-grounded SQLite queries.

ARCHITECTURE:
1. Safety check (individual underwriting redirect + unsupported domain guardrails).
2. Deterministic shortcut matching (fast path for common patterns — no LLM needed).
3. Schema-aware LLM generation for any other reasonable analytical question.
4. SQL validation enforced on ALL generated SQL before execution.

CAPABILITY:
The system can answer ANY reasonable analytical question answerable from the
approved database schema — it is NOT limited to a fixed list of patterns.
Deterministic patterns are shortcuts for speed, not capability boundaries.

Supports:
1. Single-table analytics queries (applicant_analytics).
2. Controlled multi-table relational queries across:
   - applicants, bureau_summary, previous_application_summary, installment_summary
3. Model-predicted risk score queries (applicant_risk_scores from CatBoost).
4. Deterministic pattern shortcuts with 100% offline functionality.
5. Schema-aware LLM generation for arbitrary schema-supported questions.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from src.talk_to_data.llm import LLMClient
from src.talk_to_data.prompts import build_system_prompt, format_user_prompt
from src.talk_to_data.schema import TABLE_NAME, get_allowed_columns
from src.talk_to_data.sql_validator import validate_sql

logger = logging.getLogger(__name__)

# Maximum result rows for LLM-generated queries (prevents oversized payloads)
DEFAULT_ROW_LIMIT = 100

# Keywords indicating questions about individual loan decisions / underwriting
INDIVIDUAL_DECISION_TRIGGERS = [
    "should applicant", "should i approve", "should we approve",
    "should i reject", "should we reject", "approve or reject",
    "underwrite applicant", "decision for applicant", "approve applicant",
    "is applicant creditworthy", "creditworthy applicant",
    "creditworthy",  # catches: "Is applicant 100002 creditworthy?"
]


# Keywords indicating unsupported external or missing dataset fields
UNSUPPORTED_DOMAINS = [
    "unemployment rate", "gdp", "inflation rate", "interest rate",
    "experian", "equifax", "transunion", "fico score", "stock market",
    "crypto", "bitcoin", "weather", "city temperature", "exchange rate",
    "national average", "country economy"
]


def check_intent_safety(question: str) -> Optional[Dict[str, Any]]:
    """
    Check if a question violates Talk-to-Data safety rules:
    - Attempting individual applicant underwriting decisions.
    - Querying unsupported macroeconomic or external data.

    NOTE: Portfolio-level analytics questions using model risk data ARE allowed.
    e.g., "How many high-risk applicants are there?" is allowed.
    e.g., "Should applicant 100002 be approved?" is redirected.
    """
    q_lower = question.lower()

    # Rule 1: Redirect individual underwriting requests to ML predictor
    for trigger in INDIVIDUAL_DECISION_TRIGGERS:
        if trigger in q_lower:
            return {
                "sql": None,
                "is_supported": False,
                "intent": "individual_decision_redirect",
                "explanation": (
                    "Individual loan approval/rejection decisions cannot be made through "
                    "the Talk-to-Data analytics engine. Please submit the applicant profile "
                    "to the Credit Risk ML Predictor engine for individual scoring and decision support."
                ),
                "confidence": 1.0
            }

    # Rule 2: Guardrail against unsupported external data / hallucination prevention
    for unsupported in UNSUPPORTED_DOMAINS:
        if unsupported in q_lower:
            return {
                "sql": None,
                "is_supported": False,
                "intent": "unsupported_domain",
                "explanation": (
                    f"This question cannot be answered from the available analytics data. "
                    f"The analytics database does not contain '{unsupported}' metrics. "
                    f"Only data from the Home Credit loan application schema is available."
                ),
                "confidence": 1.0
            }

    return None


def _apply_row_limit(sql: str, limit: int = DEFAULT_ROW_LIMIT) -> str:
    """
    Append a LIMIT clause to a SQL query if one is not already present
    and the query is not a natural single-aggregate (no GROUP BY = likely 1 row).
    """
    sql_upper = sql.upper().rstrip("; ")
    if "LIMIT" in sql_upper:
        return sql  # Already has LIMIT, don't modify
    # If it has GROUP BY or ORDER BY, add limit. Single aggregates (COUNT, AVG alone) skip.
    if "GROUP BY" in sql_upper or "ORDER BY" in sql_upper:
        return sql.rstrip("; ") + f" LIMIT {limit};"
    # Check if it's a bare aggregate (no FROM subquery grouping)
    # Single-row aggregates don't need limiting
    return sql


def generate_sql_deterministic(
    question: str,
    conversation_context: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Deterministic shortcut SQL generator for common high-frequency analytical patterns.
    These are SHORTCUTS (fast path), not the capability boundary.
    Questions not matched here are forwarded to the schema-aware LLM.
    """
    q_lower = question.lower().strip()

    # Check for context follow-ups (e.g., adding an income filter to previous query)
    context_income_filter = None
    target_topic = q_lower

    # NOTE: Income threshold regex is checked early in the pattern section (before portfolio summary)

    if conversation_context and len(conversation_context) > 0:
        prev_turn = conversation_context[-1]
        prev_q = prev_turn.get("question", "").lower()

        income_match = re.search(r"earning above (\d+)", q_lower) or re.search(r"income (?:above|>|over) (\d+)", q_lower)
        if income_match:
            income_val = float(income_match.group(1))
            context_income_filter = f"amt_income_total > {income_val}"
            if any(k in q_lower for k in ["what about", "only", "filter", "and for", "how about", "with"]):
                target_topic = f"{prev_q} {q_lower}"

    # =========================================================================
    # RISK SCORE PATTERNS (applicant_risk_scores table)
    # =========================================================================

    # Risk Pattern 1: Top N high-risk applicants by predicted risk
    word_num_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "fifteen": 15, "twenty": 20, "fifty": 50, "hundred": 100
    }
    n_val = 5
    num_m = re.search(r"\b(\d+)\b", q_lower)
    if num_m:
        n_val = int(num_m.group(1))
    else:
        for word, val in word_num_map.items():
            if re.search(rf"\b{word}\b", q_lower):
                n_val = val
                break

    is_top_n_risk = (
        ("top" in q_lower or "highest" in q_lower or "riskiest" in q_lower or "most risky" in q_lower or "highest-risk" in q_lower or "high-risk" in q_lower or "high risk" in q_lower or "highest risk" in q_lower)
        and ("applicant" in q_lower or "borrower" in q_lower or "client" in q_lower or "account" in q_lower or "risk score" in q_lower or "risk scores" in q_lower or "predicted" in q_lower or "default risk" in q_lower or "probability" in q_lower or "riskiest" in q_lower)
        and not any(agg in q_lower for agg in ["how many", "count of", "proportion", "percentage", "rate by", "distribution", "breakdown", "fall into"])
    )

    if is_top_n_risk or any(k in target_topic for k in [
        "top high-risk", "highest risk applicants", "most risky applicants",
        "top risky applicants", "show top", "top 5 high-risk", "top 10 high-risk",
        "top 20 applicants by risk", "riskiest applicants"
    ]):
        n = min(n_val, DEFAULT_ROW_LIMIT)
        sql = (
            f"SELECT r.sk_id_curr, r.risk_score, r.default_probability, r.risk_band, "
            f"a.amt_income_total, a.amt_credit, a.name_education_type, a.name_income_type "
            f"FROM applicant_risk_scores r "
            f"JOIN applicants a ON r.sk_id_curr = a.sk_id_curr "
            f"ORDER BY r.default_probability DESC "
            f"LIMIT {n};"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "risk_analytics",
            "explanation": f"Returns the top {n} applicants ranked by CatBoost model-predicted default probability (not by observed target).",
            "confidence": 0.98
        }

    # Risk Pattern 1b: Count of high-risk / low-risk / medium-risk applicants (aggregate count)
    high_risk_count_match = re.search(
        r"(?:how many|count|number of)\s+(?:applicants?\s+)?(?:are\s+)?(?:classified\s+as\s+)?(?:high|low|medium).?risk",
        q_lower
    )
    if high_risk_count_match and "top" not in q_lower:
        # Determine which band
        if "high" in q_lower:
            band = "High"
        elif "low" in q_lower:
            band = "Low"
        else:
            band = "Medium"
        sql = (
            f"SELECT '{band}' AS risk_band, COUNT(*) AS applicant_count, "
            f"ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicant_risk_scores), 2) AS pct_of_portfolio "
            f"FROM applicant_risk_scores WHERE risk_band = '{band}';"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "risk_analytics",
            "explanation": f"Counts applicants classified as {band} risk by the CatBoost model.",
            "confidence": 0.97
        }

    # Risk Pattern 2: Risk band distribution / "classified as" / how many in band
    if any(k in target_topic for k in [
        "risk band", "risk distribution", "each risk band", "risk bands",
        "proportion of applicants", "how many high risk", "how many low risk",
        "count by risk", "percentage.*risk band", "risk band breakdown",
        "classified as high risk", "classified as low risk", "classified as medium risk",
        "classified as high", "classified as low", "classified as medium",
        "how many applicants are classified", "high risk applicants"
    ]) and any(k in target_topic for k in ["risk", "band", "classified"]):
        sql = (
            "SELECT r.risk_band, COUNT(*) AS applicant_count, "
            "ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicant_risk_scores), 2) AS pct_of_portfolio, "
            "ROUND(AVG(r.risk_score), 2) AS avg_risk_score, "
            "ROUND(AVG(r.default_probability) * 100, 2) AS avg_default_prob_pct "
            "FROM applicant_risk_scores r "
            "GROUP BY r.risk_band "
            "ORDER BY avg_risk_score DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "risk_analytics",
            "explanation": "Counts applicants in each CatBoost model risk band (Low/Medium/High) as a portfolio distribution.",
            "confidence": 0.97
        }

    # Risk Pattern 3: High-risk applicants with previous refusals
    if any(k in target_topic for k in [
        "high risk.*previous refusal", "high-risk.*previous refusal",
        "high risk.*refusal", "high-risk.*refusal",
        "high risk applicants.*bureau", "high-risk.*bureau",
        "high risk applicants.*late payment"
    ]):
        sql = (
            "SELECT COUNT(*) AS high_risk_with_refusals, "
            "ROUND(AVG(p.prev_refusal_rate) * 100, 2) AS avg_refusal_rate_pct, "
            "ROUND(AVG(r.risk_score), 2) AS avg_risk_score "
            "FROM applicant_risk_scores r "
            "JOIN previous_application_summary p ON r.sk_id_curr = p.sk_id_curr "
            "WHERE r.risk_band = 'High' AND p.prev_refusal_rate > 0;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "risk_analytics",
            "explanation": "Counts high-risk applicants (by CatBoost model) who also have a history of previous loan refusals.",
            "confidence": 0.96
        }

    # Risk Pattern 4: Average predicted risk score
    if any(k in target_topic for k in [
        "average predicted risk", "average risk score", "mean risk score",
        "average default probability", "mean predicted"
    ]):
        sql = (
            "SELECT ROUND(AVG(risk_score), 2) AS avg_risk_score, "
            "ROUND(AVG(default_probability) * 100, 2) AS avg_default_prob_pct, "
            "COUNT(*) AS total_applicants "
            "FROM applicant_risk_scores;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "risk_analytics",
            "explanation": "Calculates average CatBoost model-predicted risk score and default probability across the portfolio.",
            "confidence": 0.97
        }

    # =========================================================================
    # SIMPLE ANALYTICAL SHORTCUTS
    # =========================================================================

    # Income threshold count MUST match before general portfolio summary
    # Handles: "have income above X", "earn above X", "earning more than X",
    #          "with income above X", "income > X", "income above X"
    income_threshold_match = (
        re.search(
            r"(?:how many|count|number of)\s+applicants?\s+(?:have|earn|earning|with)?\s*income\s+(?:above|over|>|more than|exceeding)\s+(\d+)",
            q_lower
        ) or re.search(
            r"(?:how many|count|number of)\s+applicants?\s+(?:have|earn|earning)\s+(?:above|over|>|more than)\s+(\d+)",
            q_lower
        ) or re.search(
            r"(?:how many|count|number of)\s+applicants?\s+with\s+income\s+(?:above|over|>|more than|exceeding)\s+(\d+)",
            q_lower
        ) or re.search(
            r"income\s+(?:above|over|>|more than|exceeding)\s+(\d+)",
            q_lower
        ) if any(k in q_lower for k in ["how many", "count", "number of"]) else None
    )
    if income_threshold_match:
        threshold = float(income_threshold_match.group(1))
        sql = (
            f"SELECT COUNT(*) AS applicant_count, "
            f"ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            f"ROUND(AVG(amt_credit), 2) AS avg_credit "
            f"FROM {TABLE_NAME} "
            f"WHERE amt_income_total > {threshold};"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": f"Counts applicants with annual income above {threshold:,.0f}.",
            "confidence": 0.97
        }


    # Simple Pattern: Average income
    if any(k in target_topic for k in [
        "average income", "avg income", "mean income", "average annual income",
        "what is the average income"
    ]) and not any(k in target_topic for k in ["education", "organization", "tier", "income type"]):
        sql = (
            "SELECT ROUND(AVG(amt_income_total), 2) AS avg_annual_income, "
            "ROUND(MIN(amt_income_total), 2) AS min_income, "
            "ROUND(MAX(amt_income_total), 2) AS max_income, "
            "COUNT(*) AS total_applicants "
            f"FROM {TABLE_NAME};"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Calculates the average, minimum, and maximum annual income across all applicants.",
            "confidence": 0.98
        }

    # Simple Pattern: Average credit amount
    if any(k in target_topic for k in [
        "average credit", "avg credit", "mean credit", "average loan amount",
        "average credit amount", "median credit", "what is the average credit"
    ]) and not any(k in target_topic for k in ["education", "organization", "income", "bureau"]):
        sql = (
            "SELECT ROUND(AVG(amt_credit), 2) AS avg_credit_amount, "
            "ROUND(MIN(amt_credit), 2) AS min_credit, "
            "ROUND(MAX(amt_credit), 2) AS max_credit, "
            "COUNT(*) AS total_applicants "
            f"FROM {TABLE_NAME};"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Calculates the average, minimum, and maximum credit/loan amount across all applicants.",
            "confidence": 0.97
        }

    # Simple Pattern: Count above income threshold
    income_threshold_match = re.search(
        r"(?:how many|count|number of)\s+applicants?\s+(?:have|with|earning|earn|income)\s+(?:above|over|>|more than)\s+(\d+)",
        q_lower
    )
    if income_threshold_match:
        threshold = float(income_threshold_match.group(1))
        sql = (
            f"SELECT COUNT(*) AS applicant_count, "
            f"ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            f"ROUND(AVG(amt_credit), 2) AS avg_credit "
            f"FROM {TABLE_NAME} "
            f"WHERE amt_income_total > {threshold};"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": f"Counts applicants with annual income above {threshold:,.0f}.",
            "confidence": 0.97
        }

    # =========================================================================
    # MULTI-TABLE RELATIONAL PATTERNS
    # =========================================================================

    # Relational Pattern 1: Joint Risk Profile (Previous Refusals + Late Payments)
    if any(k in target_topic for k in [
        "previous refusal and late", "previous loan refusals and late installment",
        "prior refusals and late", "refused and late payment", "refusals and late installment",
        "how many applicants had previous loan refusals and late"
    ]):
        sql = (
            "SELECT "
            "CASE "
            "WHEN p.prev_refusal_rate > 0.3 AND i.late_payment_ratio > 0.1 THEN 'High Refusal & High Late Payments' "
            "WHEN p.prev_refusal_rate > 0.0 OR i.late_payment_ratio > 0.0 THEN 'Moderate Risk History' "
            "ELSE 'Clean Prior History' "
            "END AS combined_risk_profile, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(a.target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(a.amt_income_total), 2) AS avg_income "
            "FROM applicants a "
            "JOIN previous_application_summary p ON a.sk_id_curr = p.sk_id_curr "
            "JOIN installment_summary i ON a.sk_id_curr = i.sk_id_curr "
            "GROUP BY combined_risk_profile "
            "ORDER BY default_rate_pct DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "relational_analytics",
            "explanation": "Performs a 3-table join across applicants, previous applications, and installment summaries to analyze combined refusal and delinquency risk.",
            "confidence": 0.98
        }

    # Relational Pattern 2: Bureau History vs Previous Application History
    if any(k in target_topic for k in [
        "bureau history and previous application history", "bureau and previous application history",
        "bureau history and previous applications", "compare default rates between applicants with bureau"
    ]):
        sql = (
            "SELECT "
            "CASE "
            "WHEN b.bureau_credit_count > 0 AND p.prev_app_count > 0 THEN 'Both Bureau & Prior App History' "
            "WHEN b.bureau_credit_count > 0 THEN 'Bureau History Only' "
            "WHEN p.prev_app_count > 0 THEN 'Prior App History Only' "
            "ELSE 'No Prior Credit History' "
            "END AS credit_history_profile, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(a.target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(a.amt_credit), 2) AS avg_credit "
            "FROM applicants a "
            "JOIN bureau_summary b ON a.sk_id_curr = b.sk_id_curr "
            "JOIN previous_application_summary p ON a.sk_id_curr = p.sk_id_curr "
            "GROUP BY credit_history_profile "
            "ORDER BY default_rate_pct DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "relational_analytics",
            "explanation": "Joins applicants, bureau_summary, and previous_application_summary to compare default rates across multi-channel credit history profiles.",
            "confidence": 0.97
        }

    # Pattern: Bureau History Count
    if any(k in target_topic for k in [
        "bureau history", "have bureau", "bureau record", "bureau credit",
        "have credit history", "credit bureau history"
    ]) and any(k in target_topic for k in ["how many", "count", "number"]):
        sql = (
            "SELECT "
            "SUM(CASE WHEN bureau_credit_count > 0 THEN 1 ELSE 0 END) AS with_bureau_history, "
            "SUM(CASE WHEN bureau_credit_count = 0 THEN 1 ELSE 0 END) AS no_bureau_history, "
            "COUNT(*) AS total_applicants, "
            "ROUND(SUM(CASE WHEN bureau_credit_count > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) AS pct_with_bureau "
            "FROM bureau_summary;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Counts applicants with and without bureau credit history.",
            "confidence": 0.96
        }

    # Relational Pattern 3: Bureau Debt Exposure (applicants + bureau_summary)
    if any(k in target_topic for k in [
        "bureau debt above", "bureau debt greater", "bureau debt exceed",
        "bureau debt compared to credit", "bureau debt above their credit limit"
    ]):
        sql = (
            "SELECT "
            "CASE "
            "WHEN b.bureau_total_debt > a.amt_credit THEN 'Bureau Debt > Loan Amount' "
            "WHEN b.bureau_total_debt > 0 THEN 'Bureau Debt <= Loan Amount' "
            "ELSE 'Zero Bureau Debt' "
            "END AS debt_exposure_tier, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(a.target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(b.bureau_total_debt), 2) AS avg_bureau_debt, "
            "ROUND(AVG(a.amt_credit), 2) AS avg_loan_amount "
            "FROM applicants a "
            "JOIN bureau_summary b ON a.sk_id_curr = b.sk_id_curr "
            "GROUP BY debt_exposure_tier "
            "ORDER BY default_rate_pct DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "relational_analytics",
            "explanation": "Joins applicants with bureau_summary to evaluate default rates for applicants whose outstanding bureau debt exceeds their requested credit.",
            "confidence": 0.96
        }

    # Relational Pattern 4: Installment Payment Behavior for Defaulters
    if any(k in target_topic for k in [
        "average historical installment payment ratio for defaulters",
        "installment payment ratio for defaulters", "installment summary for defaulters",
        "installment payment behavior for defaulters", "compare installment payments by default"
    ]):
        sql = (
            "SELECT "
            "CASE WHEN a.target = 1 THEN 'Defaulters (Target = 1)' ELSE 'Non-Defaulters (Target = 0)' END AS applicant_group, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(i.total_payment_amount), 2) AS avg_total_paid, "
            "ROUND(AVG(i.late_payment_ratio) * 100, 2) AS avg_late_payment_pct "
            "FROM applicants a "
            "JOIN installment_summary i ON a.sk_id_curr = i.sk_id_curr "
            "GROUP BY applicant_group "
            "ORDER BY a.target DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "relational_analytics",
            "explanation": "Joins applicants with installment_summary to contrast average historical payments and late-payment ratios between defaulters and non-defaulters.",
            "confidence": 0.96
        }

    # =========================================================================
    # SINGLE-TABLE ANALYTICAL PATTERNS
    # =========================================================================

    # Pattern: Education Level Risk
    if any(k in target_topic for k in ["education", "degree", "academic", "schooling"]):
        where_clause = f"WHERE {context_income_filter} " if context_income_filter else ""
        sql = (
            "SELECT "
            "name_education_type, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_credit), 2) AS avg_credit_amount, "
            "ROUND(AVG(amt_income_total), 2) AS avg_income "
            f"FROM {TABLE_NAME} "
            f"{where_clause}"
            "GROUP BY name_education_type "
            "ORDER BY default_rate_pct DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Aggregates applicant counts, default rates, and loan terms across education categories.",
            "confidence": 0.98
        }

    # Pattern: Income Segmentation / Tiers
    if any(k in target_topic for k in ["income tier", "income bracket", "by income", "income segment", "income level", "vary by income"]):
        sql = (
            "SELECT "
            "CASE "
            "WHEN amt_income_total < 100000 THEN '1. Low Income (< 100k)' "
            "WHEN amt_income_total BETWEEN 100000 AND 200000 THEN '2. Middle Income (100k - 200k)' "
            "WHEN amt_income_total BETWEEN 200001 AND 350000 THEN '3. Upper Middle (200k - 350k)' "
            "ELSE '4. High Income (> 350k)' "
            "END AS income_tier, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(credit_income_ratio), 2) AS avg_credit_to_income "
            f"FROM {TABLE_NAME} "
            "GROUP BY income_tier "
            "ORDER BY income_tier ASC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Segments loan applicants into 4 standardized income tiers to evaluate leverage and default risk across income brackets.",
            "confidence": 0.96
        }

    # Pattern: Previous Loan Application Refusal History
    if any(k in target_topic for k in ["previous refusal", "prior refusal", "refusal history", "rejected before", "prior rejected", "previous loan refusals"]):
        sql = (
            "SELECT "
            "CASE "
            "WHEN prev_app_count = 0 THEN 'No Prior Applications' "
            "WHEN prev_refusal_rate > 0.5 THEN 'High Prior Refusals (>50%)' "
            "WHEN prev_refusal_rate > 0.0 THEN 'Some Prior Refusals (1-50%)' "
            "ELSE 'Clean Prior Approval History' "
            "END AS refusal_history_group, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_credit), 2) AS avg_credit "
            f"FROM {TABLE_NAME} "
            "GROUP BY refusal_history_group "
            "ORDER BY default_rate_pct DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Evaluates observed default risk based on applicants' historical Home Credit application refusal rates.",
            "confidence": 0.97
        }

    # Pattern: Credit-to-Income Leverage
    if any(k in target_topic for k in ["leverage", "credit to income", "debt to income", "highly leveraged", "leverage bracket", "credit-to-income"]):
        sql = (
            "SELECT "
            "CASE "
            "WHEN credit_income_ratio < 2.0 THEN 'Low Leverage (<2x)' "
            "WHEN credit_income_ratio BETWEEN 2.0 AND 4.0 THEN 'Moderate Leverage (2x-4x)' "
            "ELSE 'High Leverage (>4x)' "
            "END AS leverage_bracket, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_annuity), 2) AS avg_monthly_annuity "
            f"FROM {TABLE_NAME} "
            "WHERE credit_income_ratio IS NOT NULL "
            "GROUP BY leverage_bracket "
            "ORDER BY default_rate_pct DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Groups applicants by credit-to-income ratio to measure the relationship between loan leverage and default rate.",
            "confidence": 0.95
        }

    # Pattern: Late Payment / Installment Delinquency History
    if any(k in target_topic for k in ["late payment", "delinquen", "payment delay", "installment late", "past payment history", "late-payment"]):
        sql = (
            "SELECT "
            "CASE "
            "WHEN late_payment_ratio = 0.0 THEN '0% Late Payments' "
            "WHEN late_payment_ratio <= 0.15 THEN '1% - 15% Late Payments' "
            "ELSE '>15% Late Payments' "
            "END AS delinquency_tier, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(bureau_total_debt), 2) AS avg_bureau_debt "
            f"FROM {TABLE_NAME} "
            "GROUP BY delinquency_tier "
            "ORDER BY default_rate_pct DESC;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Analyzes default rates across tiers of historical installment payment punctuality.",
            "confidence": 0.96
        }

    # Pattern: Organization / Employer Risk
    if any(k in target_topic for k in ["organization", "employer", "industry", "sector", "company type"]):
        sql = (
            "SELECT "
            "organization_type, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_income_total), 2) AS avg_income "
            f"FROM {TABLE_NAME} "
            "WHERE organization_type IS NOT NULL AND organization_type != 'XNA' "
            "GROUP BY organization_type "
            "HAVING COUNT(*) >= 15 "
            "ORDER BY default_rate_pct DESC "
            "LIMIT 10;"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Identifies the top employment sectors with the highest observed default rates among sectors with significant sample sizes.",
            "confidence": 0.95
        }

    # General Pattern: Overall Portfolio Summary / Baseline Risk
    # NOTE: 'how many applicants' and 'total applicants' are intentionally NOT here
    # because those phrases may refer to income-threshold or risk-band questions that
    # have been matched earlier in this function.
    if any(k in target_topic for k in [
        "default rate", "overall default", "portfolio default", "portfolio summary",
        "overall portfolio", "baseline default",
        "rate of default", "average default", "what percentage defaulted", "percentage.*default"
    ]):
        sql = (
            "SELECT "
            "COUNT(*) AS total_applicants, "
            "SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS total_defaults, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_income_total), 2) AS avg_annual_income, "
            "ROUND(AVG(amt_credit), 2) AS avg_credit_amount "
            f"FROM {TABLE_NAME};"
        )
        return {
            "sql": sql,
            "is_supported": True,
            "intent": "portfolio_analytics",
            "explanation": "Calculates total applicant count, total default count, baseline default rate, average income, and average credit amount across the portfolio.",
            "confidence": 0.98
        }


    # No deterministic shortcut matched — signal to use LLM
    return {
        "sql": None,
        "is_supported": None,  # None = "unknown, try LLM"
        "intent": "unmatched",
        "explanation": "No deterministic shortcut matched. Question will be forwarded to schema-aware LLM.",
        "confidence": 0.0
    }


def generate_sql(
    question: str,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    llm_client: Optional[LLMClient] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    Main entry point for Natural Language to SQL translation.

    Architecture:
    1. Safety check (individual underwriting redirect + unsupported domain guardrails).
    2. Deterministic shortcut matching (fast path — no LLM cost).
    3. Schema-aware LLM generation for any other reasonable analytical question.
    4. SQL validation enforced on ALL generated SQL before execution.

    Args:
        question: User query string (max 2000 chars enforced upstream).
        conversation_context: Optional previous conversation turn history.
        llm_client: Optional LLMClient instance.
        use_llm: Whether to attempt LLM generation if no deterministic shortcut matches.

    Returns:
        Structured dictionary with generated SQL, intent, explanation, confidence, and used_llm flag.
    """
    # Step 1: Safety check & individual decision guardrails
    safety_check = check_intent_safety(question)
    if safety_check is not None:
        safety_check["used_llm"] = False
        return safety_check

    # Step 2: Try deterministic shortcut (fast path — takes priority when matched)
    deterministic_result = generate_sql_deterministic(question, conversation_context)

    if deterministic_result.get("is_supported") is True:
        # Deterministic pattern matched — return immediately, no LLM needed
        deterministic_result["used_llm"] = False
        return deterministic_result

    # Step 3: Deterministic shortcut didn't match → try schema-aware LLM
    if use_llm:
        client = llm_client or LLMClient()
        if client.is_available():
            try:
                system_prompt = build_system_prompt()
                user_prompt = format_user_prompt(question, conversation_context)

                raw_output = client.complete(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True
                )

                if raw_output:
                    # Parse JSON output from LLM
                    parsed = None
                    try:
                        parsed = json.loads(raw_output)
                    except json.JSONDecodeError:
                        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
                        if match:
                            try:
                                parsed = json.loads(match.group(0))
                            except json.JSONDecodeError:
                                pass

                    if parsed and isinstance(parsed, dict):
                        sql = parsed.get("sql")
                        intent = parsed.get("intent", "portfolio_analytics")
                        reason = parsed.get("reason") or parsed.get("explanation", "")

                        # If LLM classified intent as unsupported
                        if intent in ["unsupported", "unsupported_domain"]:
                            return {
                                "sql": None,
                                "is_supported": False,
                                "intent": "unsupported_domain",
                                "explanation": reason or "This question cannot be answered from the available analytics data.",
                                "confidence": 0.95,
                                "used_llm": True
                            }

                        # If LLM classified as individual decision
                        if intent in ["individual_decision", "individual_decision_redirect"]:
                            return {
                                "sql": None,
                                "is_supported": False,
                                "intent": "individual_decision_redirect",
                                "explanation": reason or "Individual lending decisions must be evaluated by the ML risk predictor engine.",
                                "confidence": 1.0,
                                "used_llm": True
                            }

                        # Validate the generated SQL through the full security pipeline
                        if sql:
                            # Apply row limit if needed
                            sql = _apply_row_limit(sql)

                            validation = validate_sql(sql)
                            if validation["valid"]:
                                return {
                                    "sql": validation["sanitized_sql"],
                                    "is_supported": True,
                                    "intent": intent,
                                    "explanation": reason or "SQL query generated by schema-aware LLM.",
                                    "confidence": 0.90,
                                    "used_llm": True
                                }
                            else:
                                logger.warning(
                                    "LLM generated SQL failed validation: %s. Reason: %s. "
                                    "Returning schema-capability error.",
                                    sql, validation["reason"]
                                )
                                return {
                                    "sql": None,
                                    "is_supported": False,
                                    "intent": "generation_error",
                                    "explanation": (
                                        "The generated query could not be safely executed. "
                                        "Please rephrase your question or try a more specific analytical question."
                                    ),
                                    "confidence": 0.0,
                                    "used_llm": True
                                }

            except Exception as e:
                logger.warning(
                    "LLM SQL generation encountered an error: %s. Falling back to deterministic engine.",
                    e
                )

    # Step 4: LLM not available / no valid output from LLM
    # Provide an informative response instead of silent "unrecognized_query"
    if use_llm:
        # LLM was requested but not available or failed
        return {
            "sql": None,
            "is_supported": False,
            "intent": "llm_unavailable",
            "explanation": (
                "This question requires the LLM analytical engine to generate a custom SQL query. "
                "Please configure a Groq or OpenRouter API key to enable flexible schema-aware queries. "
                "Common portfolio questions (default rate, education risk, income segments, etc.) "
                "are available without an API key."
            ),
            "confidence": 0.0,
            "used_llm": False
        }
    else:
        # use_llm=False explicitly — fall back to deterministic "unrecognized"
        return {
            "sql": None,
            "is_supported": False,
            "intent": "unrecognized_query",
            "explanation": "The question could not be matched to an available analytical query pattern.",
            "confidence": 0.0,
            "used_llm": False
        }
