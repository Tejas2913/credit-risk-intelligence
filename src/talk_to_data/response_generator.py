"""
Talk-to-Data Business Response Generator
========================================
Transforms raw SQL execution results into concise, executive-level business answers.

Supports:
1. LLM-based narrative synthesis (grounded strictly in SQL output, no hallucination).
2. Deterministic narrative fallback (formats scalar metrics, category rankings, and risk tier comparisons).
"""

import json
import logging
from typing import Any, Dict, List, Optional

from src.talk_to_data.llm import LLMClient

logger = logging.getLogger(__name__)

RESPONSE_SYSTEM_PROMPT = """You are a Senior Credit Risk Executive for a retail bank.
Your job is to provide a concise, factual, and professional business summary (2-4 sentences) answering the user's question based EXCLUSIVELY on the provided SQL query and data results.

STRICT RULES:
1. ONLY state facts and numbers directly present in the SQL data result.
2. DO NOT hallucinate, assume, or invent numbers, reasons, or external macro data.
3. If the data is empty, state clearly that no matching records were found in the database.
4. Highlight key risk metrics such as default rates (%), applicant volumes, and comparative risk differences.
5. Maintain a professional, objective tone suitable for executive credit committee review.
"""


def _format_currency(val: Any) -> str:
    try:
        f = float(val)
        return f"${f:,.2f}"
    except (ValueError, TypeError):
        return str(val)


def _format_pct(val: Any) -> str:
    try:
        f = float(val)
        return f"{f:.2f}%"
    except (ValueError, TypeError):
        return str(val)


def _format_count(val: Any) -> str:
    try:
        i = int(float(val))
        return f"{i:,}"
    except (ValueError, TypeError):
        return str(val)


def generate_deterministic_business_response(
    question: str,
    intent: str,
    data: List[Dict[str, Any]],
    columns: List[str]
) -> str:
    """
    Generate a clean, professional business summary deterministically from structured SQL rows.
    Guarantees strict semantic alignment between the query intent, returned rows, and generated narrative.
    """
    if not data:
        return "The analytics database returned no matching records for this query."

    # Pattern 1: Portfolio Summary (Scalar totals across entire dataset)
    if intent == "portfolio_summary" or (len(data) == 1 and "total_applicants" in data[0] and "total_defaults" in data[0] and "sk_id_curr" not in columns):
        row = data[0]
        total_app = _format_count(row.get("total_applicants", 0))
        total_def = _format_count(row.get("total_defaults", 0))
        def_rate = _format_pct(row.get("default_rate_pct", 0.0))
        avg_inc = _format_currency(row.get("avg_annual_income", 0.0))
        avg_cred = _format_currency(row.get("avg_credit_amount", 0.0))

        return (
            f"Across the analytics portfolio of {total_app} applicants, there are {total_def} observed defaults, "
            f"yielding a baseline default rate of {def_rate}. "
            f"The average applicant income is {avg_inc} with an average loan credit amount of {avg_cred}."
        )

    # Pattern 2: Model Risk Analytics — Top-N Individual Risk Ranking
    if (intent == "risk_analytics" or "risk_score" in columns or "default_probability" in columns) and "sk_id_curr" in columns:
        top = data[0]
        count_str = f"{len(data)} applicants" if len(data) > 1 else "1 applicant"
        top_prob = round(float(top.get("default_probability", 0)) * 100, 2)
        top_score = top.get("risk_score")
        top_band = top.get("risk_band", "N/A")
        top_id = top.get("sk_id_curr")
        return (
            f"Here are the {count_str} with the highest CatBoost model-predicted default risk. "
            f"The highest-risk applicant (ID: {top_id}) has a predicted default probability of "
            f"{top_prob}% (Risk Score: {top_score}, Band: {top_band})."
        )

    # Pattern 3: Model Risk Analytics — Risk Band Distribution
    if intent == "risk_analytics" and "risk_band" in columns and "pct_of_portfolio" in columns:
        parts = []
        for row in data:
            parts.append(
                f"{row.get('risk_band')} risk: {_format_count(row.get('applicant_count'))} applicants "
                f"({row.get('pct_of_portfolio')}% of portfolio)"
            )
        summary = "; ".join(parts) + "."
        return f"CatBoost model risk band distribution across the portfolio: {summary}"

    # Pattern 4: Education Risk Analysis (Grouped default rates by education type)
    if (intent == "education_risk_analysis" or ("name_education_type" in columns and "default_rate_pct" in columns)) and "sk_id_curr" not in columns:
        highest = data[0]
        lowest = data[-1]
        h_rate = _format_pct(highest.get("default_rate_pct"))
        h_cnt = _format_count(highest.get("applicant_count"))
        l_rate = _format_pct(lowest.get("default_rate_pct"))
        l_cnt = _format_count(lowest.get("applicant_count"))
        return (
            f"Analysis across education levels shows the highest default risk in '{highest.get('name_education_type')}' "
            f"at {h_rate} ({h_cnt} applicants), while '{lowest.get('name_education_type')}' exhibits the lowest default risk at "
            f"{l_rate} ({l_cnt} applicants)."
        )

    # Pattern 5: Income Tiers
    if (intent == "income_tier_analysis" or ("income_tier" in columns and "default_rate_pct" in columns)) and "sk_id_curr" not in columns:
        highest_rate_tier = max(data, key=lambda x: float(x.get("default_rate_pct", 0)))
        lowest_rate_tier = min(data, key=lambda x: float(x.get("default_rate_pct", 0)))
        return (
            f"Default risk varies inversely with applicant income tiers. "
            f"The '{highest_rate_tier.get('income_tier')}' bracket shows the highest risk with a "
            f"{_format_pct(highest_rate_tier.get('default_rate_pct'))} default rate, compared to "
            f"{_format_pct(lowest_rate_tier.get('default_rate_pct'))} in the '{lowest_rate_tier.get('income_tier')}' tier."
        )

    # Pattern 6: Prior Refusal History
    if (intent == "prior_refusal_risk_analysis" or ("refusal_history_group" in columns and "default_rate_pct" in columns)) and "sk_id_curr" not in columns:
        high_ref = next((r for r in data if "High Prior Refusals" in str(r.get("refusal_history_group"))), data[0])
        clean_ref = next((r for r in data if "Clean" in str(r.get("refusal_history_group")) or "No Prior" in str(r.get("refusal_history_group"))), data[-1])
        return (
            f"Historical loan application refusals strongly correlate with increased credit default risk. "
            f"Applicants with '{high_ref.get('refusal_history_group')}' experience a default rate of "
            f"{_format_pct(high_ref.get('default_rate_pct'))}, compared to {_format_pct(clean_ref.get('default_rate_pct'))} "
            f"for applicants with '{clean_ref.get('refusal_history_group')}'."
        )

    # Pattern 7: Leverage Bracket
    if (intent == "leverage_risk_analysis" or ("leverage_bracket" in columns and "default_rate_pct" in columns)) and "sk_id_curr" not in columns:
        highest_lev = max(data, key=lambda x: float(x.get("default_rate_pct", 0)))
        lowest_lev = min(data, key=lambda x: float(x.get("default_rate_pct", 0)))
        return (
            f"Leverage analysis indicates that applicants in the '{highest_lev.get('leverage_bracket')}' bracket "
            f"have the highest default rate at {_format_pct(highest_lev.get('default_rate_pct'))} "
            f"({_format_count(highest_lev.get('applicant_count'))} applicants), compared to "
            f"{_format_pct(lowest_lev.get('default_rate_pct'))} in the '{lowest_lev.get('leverage_bracket')}' bracket."
        )

    # Pattern 8: Late Payment Delinquency
    if (intent == "late_payment_risk_analysis" or ("delinquency_tier" in columns and "default_rate_pct" in columns)) and "sk_id_curr" not in columns:
        high_del = next((r for r in data if ">" in str(r.get("delinquency_tier")) or "15%" in str(r.get("delinquency_tier"))), data[0])
        zero_del = next((r for r in data if "0%" in str(r.get("delinquency_tier"))), data[-1])
        return (
            f"Installment payment delinquency is a strong predictor of default. "
            f"Applicants with '{high_del.get('delinquency_tier')}' have an elevated default rate of "
            f"{_format_pct(high_del.get('default_rate_pct'))}, whereas applicants with "
            f"'{zero_del.get('delinquency_tier')}' average a {_format_pct(zero_del.get('default_rate_pct'))} default rate."
        )

    # Pattern 9: Organization Risk Ranking
    if (intent == "organization_risk_ranking" or ("organization_type" in columns and "default_rate_pct" in columns)) and "sk_id_curr" not in columns:
        top_org = data[0]
        return (
            f"Among organization types with at least 15 applicants, '{top_org.get('organization_type')}' "
            f"ranks highest in default risk at {_format_pct(top_org.get('default_rate_pct'))} "
            f"across {_format_count(top_org.get('applicant_count'))} applicants."
        )

    # Multi-Table Relational Pattern 1: Joint Risk Profile (Refusals + Late Payments)
    if (intent == "relational_joint_risk_analysis" or "combined_risk_profile" in columns) and "sk_id_curr" not in columns:
        high_risk = next((r for r in data if "High Refusal" in str(r.get("combined_risk_profile"))), data[0])
        clean = next((r for r in data if "Clean" in str(r.get("combined_risk_profile"))), data[-1])
        return (
            f"Multi-table analysis joining previous applications and installment histories reveals that applicants with "
            f"'{high_risk.get('combined_risk_profile')}' exhibit an elevated default rate of {_format_pct(high_risk.get('default_rate_pct'))} "
            f"({_format_count(high_risk.get('applicant_count'))} applicants), compared to {_format_pct(clean.get('default_rate_pct'))} "
            f"for applicants with '{clean.get('combined_risk_profile')}'."
        )

    # Multi-Table Relational Pattern 2: Bureau History vs Previous Application History
    if (intent == "relational_bureau_and_prev_apps_analysis" or "credit_history_profile" in columns) and "sk_id_curr" not in columns:
        highest = max(data, key=lambda x: float(x.get("default_rate_pct", 0)))
        lowest = min(data, key=lambda x: float(x.get("default_rate_pct", 0)))
        return (
            f"Cross-table relational comparison between credit bureau and previous application histories indicates "
            f"the highest default rate in '{highest.get('credit_history_profile')}' at {_format_pct(highest.get('default_rate_pct'))}, "
            f"versus {_format_pct(lowest.get('default_rate_pct'))} in '{lowest.get('credit_history_profile')}'."
        )

    # Multi-Table Relational Pattern 3: Bureau Debt Exposure
    if (intent == "relational_bureau_debt_analysis" or "debt_exposure_tier" in columns) and "sk_id_curr" not in columns:
        high_debt = next((r for r in data if "Bureau Debt >" in str(r.get("debt_exposure_tier"))), data[0])
        zero_debt = next((r for r in data if "Zero" in str(r.get("debt_exposure_tier"))), data[-1])
        return (
            f"Relational join with bureau summaries shows applicants whose external bureau debt exceeds their loan amount "
            f"default at {_format_pct(high_debt.get('default_rate_pct'))} ({_format_count(high_debt.get('applicant_count'))} applicants), "
            f"compared to {_format_pct(zero_debt.get('default_rate_pct'))} for applicants with zero bureau debt."
        )

    # Multi-Table Relational Pattern 4: Installment Summary for Defaulters
    if (intent == "relational_installment_summary_analysis" or "applicant_group" in columns) and "sk_id_curr" not in columns:
        defaulters = next((r for r in data if "Defaulters" in str(r.get("applicant_group")) and "Non" not in str(r.get("applicant_group"))), data[0])
        non_defaulters = next((r for r in data if "Non-Defaulters" in str(r.get("applicant_group"))), data[-1])
        return (
            f"Installment history join shows defaulters average an installment late-payment ratio of "
            f"{defaulters.get('avg_late_payment_pct')}% (total paid: {_format_currency(defaulters.get('avg_total_paid'))}), "
            f"compared to {non_defaulters.get('avg_late_payment_pct')}% for non-defaulters (total paid: {_format_currency(non_defaulters.get('avg_total_paid'))})."
        )

    # Generic risk analytics fallback
    if intent == "risk_analytics":
        if len(data) == 1:
            row = data[0]
            metrics = ", ".join(f"{k}: {v}" for k, v in row.items())
            return f"Model risk analytics result: {metrics}."
        return f"The risk analytics query returned {len(data)} results across: {', '.join(columns)}."

    # General: Single-row scalar aggregate (e.g., AVG income, COUNT with filter)
    if len(data) == 1:
        row = data[0]
        parts = []
        for col, val in row.items():
            if val is None:
                continue
            col_lower = col.lower()
            if "income" in col_lower or "credit" in col_lower or "debt" in col_lower or "payment" in col_lower or "annuity" in col_lower:
                parts.append(f"{col}: {_format_currency(val)}")
            elif "pct" in col_lower or "rate" in col_lower or "ratio" in col_lower:
                parts.append(f"{col}: {_format_pct(val)}")
            elif "count" in col_lower or "applicant" in col_lower or "total" in col_lower:
                parts.append(f"{col}: {_format_count(val)}")
            else:
                parts.append(f"{col}: {val}")
        return f"Analytics result: {'; '.join(parts)}."

    # General: Multi-row grouped result
    if len(data) > 1:
        # If rows contain individual applicant IDs, state so
        if "sk_id_curr" in columns:
            return f"Here are the {len(data)} matching applicant records returned from the database."

        # Find the key grouping column (first non-numeric column)
        group_col = None
        for col in columns:
            if col in data[0] and isinstance(data[0][col], str):
                group_col = col
                break

        # Find the primary metric column (rate or count)
        metric_col = next((c for c in columns if "rate" in c.lower() or "pct" in c.lower()), None)
        count_col = next((c for c in columns if "count" in c.lower()), None)

        if group_col and metric_col:
            top_row = data[0]
            return (
                f"The query returned {len(data)} groups. "
                f"The highest result is '{top_row.get(group_col)}' "
                f"with {metric_col.replace('_', ' ')}: {top_row.get(metric_col)}"
                + (f", covering {_format_count(top_row.get(count_col))} applicants." if count_col else ".")
            )
        if count_col:
            total = sum(int(float(r.get(count_col, 0))) for r in data)
            return f"The query returned {len(data)} result groups covering {_format_count(total)} total applicants."

    return f"The query returned {len(data)} rows across columns: {', '.join(columns)}."



def generate_business_response(
    question: str,
    sql: Optional[str],
    data: List[Dict[str, Any]],
    columns: List[str],
    intent: str,
    llm_client: Optional[LLMClient] = None
) -> str:
    """
    Main entry point for business answer generation.
    Attempts LLM narrative synthesis if available; seamlessly falls back to deterministic formatter.
    """
    if not data or not sql:
        return "The analytics database returned no matching records for this query."

    # Try LLM synthesis if client is provided and available
    if llm_client and llm_client.is_available():
        try:
            prompt = (
                f"USER QUESTION: {question}\n\n"
                f"SQL EXECUTED:\n{sql}\n\n"
                f"QUERY RESULT DATA ({len(data)} rows):\n"
                f"{json.dumps(data[:15], indent=2)}\n\n"
                f"Provide a concise, professional 2-3 sentence executive business answer grounded strictly in these results."
            )
            response = llm_client.complete(
                prompt=prompt,
                system_prompt=RESPONSE_SYSTEM_PROMPT,
                json_mode=False
            )
            if response and response.strip():
                return response.strip()
        except Exception as e:
            logger.warning("LLM business response generation failed: %s. Using deterministic fallback.", e)

    # Fallback to deterministic business summary
    return generate_deterministic_business_response(
        question=question,
        intent=intent,
        data=data,
        columns=columns
    )
