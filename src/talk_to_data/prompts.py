"""
Talk-to-Data Prompt Engineering & Template Module
=================================================
Provides token-optimized, schema-aware prompt templates for natural-language to SQL
generation with strict hallucination controls, multi-table relational join guidance,
model-risk analytics support, and safety constraints.

DESIGN PRINCIPLE: The LLM can answer ANY reasonable analytical question answerable
from the approved database schema. It is NOT limited to a fixed list of patterns.
Deterministic patterns are fast-path shortcuts only.
"""

from typing import Any, Dict, List, Optional
from src.talk_to_data.schema import TABLE_NAME, get_schema_prompt_text

SQL_SYSTEM_PROMPT = """You are an expert Credit Risk Analytics AI Assistant for the Home Credit analytics platform.
Your task is to convert user natural-language questions into clean, secure, single-statement SQLite SELECT queries.

{schema_text}

CAPABILITY:
You can answer ANY reasonable analytical question that can be derived from the approved database schema above.
You are NOT limited to a fixed set of question types. Compose the appropriate SQL based on what the user asks.

INTENT CATEGORIES (choose the most appropriate):
- "portfolio_analytics": Single-table aggregate questions about the overall portfolio
- "relational_analytics": Multi-table join questions across applicants, bureau, previous apps, installments
- "risk_analytics": Questions using the applicant_risk_scores model predictions
- "individual_decision_redirect": User asks about approving/rejecting a specific individual applicant
- "unsupported": Question cannot be answered from the available schema (no hallucination)

ALLOWED SQL CONSTRUCTS:
- SELECT with column aliases
- WHERE with any valid column condition (=, >, <, >=, <=, BETWEEN, LIKE, IS NULL, IS NOT NULL)
- GROUP BY with HAVING
- ORDER BY ... ASC/DESC
- LIMIT N (always apply LIMIT 100 for row results unless it's a natural single-row aggregate)
- COUNT, SUM, AVG, MIN, MAX, ROUND
- CASE WHEN ... THEN ... ELSE ... END
- Approved INNER JOIN / LEFT JOIN on sk_id_curr

CRITICAL SAFETY RULES:
1. ONLY use column names explicitly listed in the schema. DO NOT hallucinate or invent columns.
2. ONLY query approved tables. NO sqlite_master, no internal tables.
3. ONLY generate a single SELECT statement. NEVER use INSERT, UPDATE, DELETE, DROP, CREATE, ALTER, PRAGMA, ATTACH.
4. If a question asks about data not in the schema (e.g., "interest rate", "Experian score", "unemployment rate"):
   Set intent to "unsupported" and explain what is missing.
5. If a question asks whether a SPECIFIC individual applicant should be approved or rejected:
   Set intent to "individual_decision_redirect". Do NOT generate SQL.
   (Portfolio analytics questions like "how many high-risk applicants are there" are ALLOWED.)
6. NEVER use the 'target' column as a substitute for predicted risk. Use applicant_risk_scores for predicted risk.
7. Output ONLY valid JSON with keys: "sql", "intent", "reason".
"""

FEW_SHOT_EXAMPLES: List[Dict[str, str]] = [
    # Example 1: Simple average
    {
        "question": "What is the average annual income of applicants?",
        "sql": (
            "SELECT ROUND(AVG(amt_income_total), 2) AS avg_annual_income, "
            "COUNT(*) AS total_applicants "
            f"FROM {TABLE_NAME};"
        ),
        "intent": "portfolio_analytics"
    },
    # Example 2: Filtered count
    {
        "question": "How many applicants have income above 300000?",
        "sql": (
            "SELECT COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_credit), 2) AS avg_credit "
            f"FROM {TABLE_NAME} "
            "WHERE amt_income_total > 300000;"
        ),
        "intent": "portfolio_analytics"
    },
    # Example 3: Overall default rate
    {
        "question": "What is the overall portfolio default rate and total applicant count?",
        "sql": (
            "SELECT COUNT(*) AS total_applicants, "
            "SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) AS total_defaults, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_income_total), 2) AS avg_annual_income, "
            "ROUND(AVG(amt_credit), 2) AS avg_credit_amount "
            f"FROM {TABLE_NAME};"
        ),
        "intent": "portfolio_analytics"
    },
    # Example 4: Grouped by education
    {
        "question": "What is the default rate by education level?",
        "sql": (
            "SELECT name_education_type, COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_credit), 2) AS avg_credit_amount, "
            "ROUND(AVG(amt_income_total), 2) AS avg_income "
            f"FROM {TABLE_NAME} "
            "GROUP BY name_education_type "
            "ORDER BY default_rate_pct DESC;"
        ),
        "intent": "portfolio_analytics"
    },
    # Example 5: Top high-risk applicants from risk scores table
    {
        "question": "Show me the top 5 highest-risk applicants by predicted risk score.",
        "sql": (
            "SELECT r.sk_id_curr, r.risk_score, r.default_probability, r.risk_band, "
            "a.amt_income_total, a.amt_credit, a.name_education_type, a.name_income_type "
            "FROM applicant_risk_scores r "
            "JOIN applicants a ON r.sk_id_curr = a.sk_id_curr "
            "ORDER BY r.default_probability DESC "
            "LIMIT 5;"
        ),
        "intent": "risk_analytics"
    },
    # Example 6: Risk band distribution
    {
        "question": "What proportion of applicants fall into each risk band?",
        "sql": (
            "SELECT r.risk_band, COUNT(*) AS applicant_count, "
            "ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM applicant_risk_scores), 2) AS pct_of_portfolio, "
            "ROUND(AVG(r.risk_score), 2) AS avg_risk_score "
            "FROM applicant_risk_scores r "
            "GROUP BY r.risk_band "
            "ORDER BY avg_risk_score DESC;"
        ),
        "intent": "risk_analytics"
    },
    # Example 7: Compound filter — income + refusal rate
    {
        "question": "What is the default rate for applicants with income above 200000 and previous refusal rate above 50%?",
        "sql": (
            "SELECT COUNT(*) AS applicant_count, "
            "ROUND(AVG(target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(amt_income_total), 2) AS avg_income, "
            "ROUND(AVG(amt_credit), 2) AS avg_credit "
            f"FROM {TABLE_NAME} "
            "WHERE amt_income_total > 200000 AND prev_refusal_rate > 0.5;"
        ),
        "intent": "portfolio_analytics"
    },
    # Example 8: Relational — high-risk + late payment
    {
        "question": "Among high-risk applicants, what is the average late-payment ratio?",
        "sql": (
            "SELECT r.risk_band, COUNT(*) AS applicant_count, "
            "ROUND(AVG(i.late_payment_ratio) * 100, 2) AS avg_late_payment_pct, "
            "ROUND(AVG(r.risk_score), 2) AS avg_risk_score "
            "FROM applicant_risk_scores r "
            "JOIN installment_summary i ON r.sk_id_curr = i.sk_id_curr "
            "WHERE r.risk_band = 'High' "
            "GROUP BY r.risk_band;"
        ),
        "intent": "risk_analytics"
    },
    # Example 9: Relational join (bureau debt vs default)
    {
        "question": "Compare default rates for applicants with bureau debt above their credit limit.",
        "sql": (
            "SELECT CASE "
            "WHEN b.bureau_total_debt > a.amt_credit THEN 'Bureau Debt > Loan Amount' "
            "WHEN b.bureau_total_debt > 0 THEN 'Bureau Debt <= Loan Amount' "
            "ELSE 'Zero Bureau Debt' END AS debt_exposure_tier, "
            "COUNT(*) AS applicant_count, "
            "ROUND(AVG(a.target) * 100, 2) AS default_rate_pct, "
            "ROUND(AVG(b.bureau_total_debt), 2) AS avg_bureau_debt "
            "FROM applicants a "
            "JOIN bureau_summary b ON a.sk_id_curr = b.sk_id_curr "
            "GROUP BY debt_exposure_tier "
            "ORDER BY default_rate_pct DESC;"
        ),
        "intent": "relational_analytics"
    },
    # Example 10: Individual decision redirect
    {
        "question": "Should applicant 100002 be approved for a loan?",
        "sql": None,
        "intent": "individual_decision_redirect"
    },
    # Example 11: Unsupported domain
    {
        "question": "What is the current interest rate in India?",
        "sql": None,
        "intent": "unsupported"
    },
]


def build_system_prompt() -> str:
    """Generate complete formatted system prompt with dynamic schema."""
    schema_text = get_schema_prompt_text()
    return SQL_SYSTEM_PROMPT.format(schema_text=schema_text)


def format_user_prompt(
    question: str,
    conversation_context: Optional[List[Dict[str, str]]] = None
) -> str:
    """Format user prompt including optional previous conversation turns for context."""
    prompt_parts = []

    if conversation_context:
        prompt_parts.append("PREVIOUS CONVERSATION CONTEXT (for follow-up questions):")
        for turn in conversation_context[-3:]:  # limit to last 3 turns for token efficiency
            prompt_parts.append(f"User: {turn.get('question', '')}")
            if turn.get('sql'):
                prompt_parts.append(f"SQL Used: {turn['sql']}")
            if turn.get('intent'):
                prompt_parts.append(f"Intent: {turn['intent']}")
        prompt_parts.append("")

    prompt_parts.append(f"CURRENT QUESTION: {question}")
    prompt_parts.append("")
    prompt_parts.append(
        "Generate the appropriate SQLite SELECT query for this question based on the approved schema. "
        "Return ONLY valid JSON with keys: \"sql\", \"intent\", \"reason\"."
    )
    return "\n".join(prompt_parts)
