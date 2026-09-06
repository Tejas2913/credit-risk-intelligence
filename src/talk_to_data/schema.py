"""
Talk-to-Data Schema Definition Module
=====================================
Authoritative schema metadata describing the analytics tables and columns
available for natural-language to SQL querying, including multi-table relational joins
and model-predicted risk score analytics.

IMPORTANT: The applicant_risk_scores table contains CatBoost model predictions.
Do NOT use the 'target' column as a substitute for predicted risk.
"""

from typing import Any, Dict, List, Set, Tuple

TABLE_NAME = "applicant_analytics"

ALLOWED_TABLES: Set[str] = {
    "applicant_analytics",
    "applicants",
    "bureau_summary",
    "previous_application_summary",
    "installment_summary",
    "applicant_risk_scores",
}

# Table-specific column schemas
TABLE_SCHEMAS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "applicants": {
        "sk_id_curr": {"type": "INTEGER", "description": "Unique identifier for the loan applicant (Primary Key)."},
        "target": {"type": "INTEGER", "description": "Observed loan default outcome: 1 = Client experienced repayment difficulties / defaulted, 0 = Non-default. NOTE: This is the observed outcome, NOT a model prediction."},
        "name_contract_type": {"type": "TEXT", "description": "Type of loan contract: 'Cash loans' or 'Revolving loans'."},
        "code_gender": {"type": "TEXT", "description": "Gender of applicant ('F', 'M', 'XNA')."},
        "flag_own_car": {"type": "TEXT", "description": "Flag indicating if applicant owns a car ('Y', 'N')."},
        "flag_own_realty": {"type": "TEXT", "description": "Flag indicating if applicant owns real estate ('Y', 'N')."},
        "cnt_children": {"type": "INTEGER", "description": "Number of children."},
        "amt_income_total": {"type": "REAL", "description": "Total annual income of the applicant in local currency."},
        "amt_credit": {"type": "REAL", "description": "Total credit/loan amount granted."},
        "amt_annuity": {"type": "REAL", "description": "Monthly loan annuity payment amount."},
        "amt_goods_price": {"type": "REAL", "description": "Price of the goods financed for consumer loans."},
        "name_income_type": {"type": "TEXT", "description": "Income source type: 'Working', 'Commercial associate', 'Pensioner', 'State servant', etc."},
        "name_education_type": {"type": "TEXT", "description": "Highest education achieved: 'Higher education', 'Secondary / secondary special', etc."},
        "name_family_status": {"type": "TEXT", "description": "Family status: 'Married', 'Single / not married', 'Civil marriage', etc."},
        "name_housing_type": {"type": "TEXT", "description": "Housing situation: 'House / apartment', 'Rented apartment', etc."},
        "age_years": {"type": "REAL", "description": "Applicant age in years."},
        "employment_years": {"type": "REAL", "description": "Employment duration in years."},
        "occupation_type": {"type": "TEXT", "description": "Occupation category (e.g. 'Laborers', 'Core staff', 'Managers')."},
        "organization_type": {"type": "TEXT", "description": "Type of employer organization (e.g. 'Business Entity Type 3', 'Self-employed')."},
        "ext_source_1": {"type": "REAL", "description": "External credit rating score 1 (0.0 to 1.0, higher means lower risk)."},
        "ext_source_2": {"type": "REAL", "description": "External credit rating score 2 (0.0 to 1.0, higher means lower risk)."},
        "ext_source_3": {"type": "REAL", "description": "External credit rating score 3 (0.0 to 1.0, higher means lower risk)."},
        "credit_income_ratio": {"type": "REAL", "description": "Ratio of credit loan amount to annual income (amt_credit / amt_income_total)."},
        "annuity_income_ratio": {"type": "REAL", "description": "Ratio of monthly annuity to income (amt_annuity / amt_income_total)."},
        "credit_goods_ratio": {"type": "REAL", "description": "Ratio of credit amount to goods price (amt_credit / amt_goods_price)."}
    },
    "bureau_summary": {
        "sk_id_curr": {"type": "INTEGER", "description": "Unique applicant ID (Primary Key / Foreign Key to applicants.sk_id_curr)."},
        "bureau_credit_count": {"type": "INTEGER", "description": "Total number of previous credits recorded in external Credit Bureau."},
        "bureau_total_debt": {"type": "REAL", "description": "Total outstanding debt across all external credit bureau loans."}
    },
    "previous_application_summary": {
        "sk_id_curr": {"type": "INTEGER", "description": "Unique applicant ID (Primary Key / Foreign Key to applicants.sk_id_curr)."},
        "prev_app_count": {"type": "INTEGER", "description": "Total number of previous loan applications submitted to Home Credit."},
        "prev_refusal_rate": {"type": "REAL", "description": "Proportion of previous Home Credit loan applications that were refused (0.0 to 1.0)."}
    },
    "installment_summary": {
        "sk_id_curr": {"type": "INTEGER", "description": "Unique applicant ID (Primary Key / Foreign Key to applicants.sk_id_curr)."},
        "total_payment_amount": {"type": "REAL", "description": "Total amount paid across all historical installment records."},
        "late_payment_ratio": {"type": "REAL", "description": "Proportion of historical installment payments made late (0.0 to 1.0)."}
    },
    "applicant_risk_scores": {
        "sk_id_curr": {"type": "INTEGER", "description": "Unique applicant ID (Primary Key / Foreign Key to applicants.sk_id_curr)."},
        "default_probability": {"type": "REAL", "description": "CatBoost model predicted probability of loan default (0.0 to 1.0). This is the ML model output, NOT the observed TARGET column."},
        "risk_score": {"type": "REAL", "description": "Scaled risk score from 0 to 100 (= default_probability * 100). Higher means higher model-predicted risk."},
        "risk_band": {"type": "TEXT", "description": "Model risk classification: 'Low' (score < 30), 'Medium' (30-60), or 'High' (score > 60). Based on CatBoost prediction thresholds."}
    },
    "applicant_analytics": {
        # Unified 31-column table containing all attributes
        "sk_id_curr": {"type": "INTEGER", "description": "Unique identifier for the loan applicant (Primary Key)."},
        "target": {"type": "INTEGER", "description": "Observed loan default outcome: 1 = Default, 0 = Non-default. NOTE: This is observed outcome, NOT model prediction."},
        "name_contract_type": {"type": "TEXT", "description": "Type of loan contract: 'Cash loans' or 'Revolving loans'."},
        "code_gender": {"type": "TEXT", "description": "Gender of applicant ('F', 'M', 'XNA')."},
        "flag_own_car": {"type": "TEXT", "description": "Car ownership ('Y', 'N')."},
        "flag_own_realty": {"type": "TEXT", "description": "Real estate ownership ('Y', 'N')."},
        "cnt_children": {"type": "INTEGER", "description": "Number of children."},
        "amt_income_total": {"type": "REAL", "description": "Total annual income."},
        "amt_credit": {"type": "REAL", "description": "Total credit/loan amount granted."},
        "amt_annuity": {"type": "REAL", "description": "Monthly loan annuity payment."},
        "amt_goods_price": {"type": "REAL", "description": "Price of goods financed."},
        "name_income_type": {"type": "TEXT", "description": "Income source type."},
        "name_education_type": {"type": "TEXT", "description": "Highest education achieved."},
        "name_family_status": {"type": "TEXT", "description": "Family status."},
        "name_housing_type": {"type": "TEXT", "description": "Housing situation."},
        "age_years": {"type": "REAL", "description": "Applicant age in years."},
        "employment_years": {"type": "REAL", "description": "Employment duration in years."},
        "occupation_type": {"type": "TEXT", "description": "Occupation category."},
        "organization_type": {"type": "TEXT", "description": "Employer organization type."},
        "ext_source_1": {"type": "REAL", "description": "External credit score 1."},
        "ext_source_2": {"type": "REAL", "description": "External credit score 2."},
        "ext_source_3": {"type": "REAL", "description": "External credit score 3."},
        "credit_income_ratio": {"type": "REAL", "description": "Credit to income ratio."},
        "annuity_income_ratio": {"type": "REAL", "description": "Annuity to income ratio."},
        "credit_goods_ratio": {"type": "REAL", "description": "Credit to goods price ratio."},
        "bureau_credit_count": {"type": "INTEGER", "description": "Credit bureau past loan count."},
        "bureau_total_debt": {"type": "REAL", "description": "Total outstanding debt in credit bureau."},
        "prev_app_count": {"type": "INTEGER", "description": "Previous Home Credit application count."},
        "prev_refusal_rate": {"type": "REAL", "description": "Previous application refusal rate."},
        "total_payment_amount": {"type": "REAL", "description": "Total installment amount paid."},
        "late_payment_ratio": {"type": "REAL", "description": "Historical late payment ratio."}
    }
}

# Legacy SCHEMA_METADATA mapping for backward compatibility
SCHEMA_METADATA = TABLE_SCHEMAS["applicant_analytics"]

# Supported join relationships (all keyed on sk_id_curr)
ALLOWED_JOIN_RELATIONSHIPS: List[Tuple[str, str, str]] = [
    ("applicants", "bureau_summary", "sk_id_curr"),
    ("applicants", "previous_application_summary", "sk_id_curr"),
    ("applicants", "installment_summary", "sk_id_curr"),
    ("applicants", "applicant_risk_scores", "sk_id_curr"),
    ("bureau_summary", "previous_application_summary", "sk_id_curr"),
    ("bureau_summary", "installment_summary", "sk_id_curr"),
    ("previous_application_summary", "installment_summary", "sk_id_curr"),
    ("applicant_risk_scores", "bureau_summary", "sk_id_curr"),
    ("applicant_risk_scores", "previous_application_summary", "sk_id_curr"),
    ("applicant_risk_scores", "installment_summary", "sk_id_curr"),
]


def get_allowed_columns(table_name: str = "applicant_analytics") -> List[str]:
    """Return list of valid column names for a given table."""
    if table_name in TABLE_SCHEMAS:
        return list(TABLE_SCHEMAS[table_name].keys())
    return list(SCHEMA_METADATA.keys())


def get_all_allowed_columns() -> Set[str]:
    """Return a set of all valid column names across all approved tables."""
    all_cols: Set[str] = set()
    for cols in TABLE_SCHEMAS.values():
        all_cols.update(cols.keys())
    return all_cols


def get_schema_prompt_text() -> str:
    """
    Format schema definition into a clean, token-efficient prompt string
    for LLM SQL generation, supporting both single-table and relational joins.

    The LLM is permitted to answer ANY reasonable analytical question
    that can be derived from the approved schema below.
    """
    lines = [
        "DATABASE SCHEMA (APPROVED ANALYTICAL TABLES):",
        "",
        "CAPABILITY STATEMENT:",
        "You can answer ANY reasonable analytical question that can be derived from the approved schema below.",
        "You are NOT limited to a fixed set of questions. Compose the appropriate SQL for the question asked.",
        "If no column exists for the requested information, set intent='unsupported' and explain why.",
        "",
        "1. TABLE 'applicants' (Core demographic, financial, and loan fields — 307,511 rows):",
        "   - sk_id_curr (INTEGER, PRIMARY KEY): Unique applicant ID",
        "   - target (INTEGER): OBSERVED loan default: 1 = Default, 0 = Non-default. NOT the model prediction.",
        "   - name_contract_type (TEXT): 'Cash loans' or 'Revolving loans'",
        "   - code_gender (TEXT): 'F', 'M', 'XNA'",
        "   - flag_own_car (TEXT): 'Y' or 'N'",
        "   - flag_own_realty (TEXT): 'Y' or 'N'",
        "   - cnt_children (INTEGER): Number of children",
        "   - amt_income_total (REAL): Annual income in local currency",
        "   - amt_credit (REAL): Total loan/credit amount granted",
        "   - amt_annuity (REAL): Monthly annuity payment",
        "   - amt_goods_price (REAL): Price of goods financed",
        "   - name_income_type (TEXT): 'Working', 'Commercial associate', 'Pensioner', 'State servant', etc.",
        "   - name_education_type (TEXT): 'Higher education', 'Secondary / secondary special', 'Incomplete higher', 'Lower secondary'",
        "   - name_family_status (TEXT): 'Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'",
        "   - name_housing_type (TEXT): 'House / apartment', 'Rented apartment', 'With parents', 'Municipal apartment'",
        "   - age_years (REAL): Applicant age in years",
        "   - employment_years (REAL): Years of employment",
        "   - occupation_type (TEXT): 'Laborers', 'Core staff', 'Managers', 'Drivers', 'Sales staff', etc.",
        "   - organization_type (TEXT): 'Business Entity Type 3', 'Self-employed', 'Government', 'School', etc.",
        "   - ext_source_1, ext_source_2, ext_source_3 (REAL): External credit scores (0.0-1.0, higher = lower risk)",
        "   - credit_income_ratio (REAL): amt_credit / amt_income_total",
        "   - annuity_income_ratio (REAL): amt_annuity / amt_income_total",
        "   - credit_goods_ratio (REAL): amt_credit / amt_goods_price",
        "",
        "2. TABLE 'bureau_summary' (Credit bureau historical metrics — one row per applicant):",
        "   - sk_id_curr (INTEGER, FK -> applicants.sk_id_curr)",
        "   - bureau_credit_count (INTEGER): Number of previous credits in external Credit Bureau (0 means no bureau history)",
        "   - bureau_total_debt (REAL): Total outstanding debt in Credit Bureau (0 means no debt recorded)",
        "",
        "3. TABLE 'previous_application_summary' (Prior Home Credit loan application history):",
        "   - sk_id_curr (INTEGER, FK -> applicants.sk_id_curr)",
        "   - prev_app_count (INTEGER): Total previous loan applications to Home Credit (0 = no prior applications)",
        "   - prev_refusal_rate (REAL): Proportion of previous applications refused (0.0 to 1.0)",
        "",
        "4. TABLE 'installment_summary' (Historical installment payment behavior):",
        "   - sk_id_curr (INTEGER, FK -> applicants.sk_id_curr)",
        "   - total_payment_amount (REAL): Total amount paid across all historical installments",
        "   - late_payment_ratio (REAL): Proportion of installment payments made late (0.0 to 1.0)",
        "",
        "5. TABLE 'applicant_risk_scores' (CatBoost ML model predicted risk — NOT the target column):",
        "   - sk_id_curr (INTEGER, FK -> applicants.sk_id_curr)",
        "   - default_probability (REAL): Model predicted probability of default (0.0 to 1.0)",
        "   - risk_score (REAL): Scaled risk score 0-100 (= default_probability * 100)",
        "   - risk_band (TEXT): 'Low' (prob < 0.30), 'Medium' (0.30-0.60), 'High' (> 0.60)",
        "   NOTE: Use this table for questions about 'high-risk applicants', 'predicted risk', 'risk score', 'risk band'.",
        "   Never use 'target' as a substitute for predicted risk.",
        "",
        "6. TABLE 'applicant_analytics' (Unified 31-column table combining all fields — use for simple single-table queries).",
        "   Same columns as 'applicants' PLUS bureau_credit_count, bureau_total_debt, prev_app_count, prev_refusal_rate,",
        "   total_payment_amount, late_payment_ratio.",
        "",
        "APPROVED RELATIONAL JOINS (all keyed on sk_id_curr):",
        "  applicants <-> bureau_summary ON sk_id_curr",
        "  applicants <-> previous_application_summary ON sk_id_curr",
        "  applicants <-> installment_summary ON sk_id_curr",
        "  applicants <-> applicant_risk_scores ON sk_id_curr",
        "  applicant_risk_scores <-> bureau_summary ON sk_id_curr",
        "  applicant_risk_scores <-> previous_application_summary ON sk_id_curr",
        "  applicant_risk_scores <-> installment_summary ON sk_id_curr",
        "",
        "IMPORTANT BUSINESS DEFINITIONS:",
        "  - 'default rate' = ROUND(AVG(target) * 100, 2) using the 'target' column (observed outcome)",
        "  - 'predicted risk' = default_probability or risk_score from applicant_risk_scores (model output)",
        "  - 'high-risk applicants' = WHERE r.risk_band = 'High' in applicant_risk_scores",
        "  - 'bureau history' = bureau_credit_count > 0",
        "  - 'previous application history' = prev_app_count > 0",
        "  - For income comparisons: use amt_income_total column",
        "  - For credit comparisons: use amt_credit column",
        "",
        "STRICT SQLITE RULES:",
        "1. Only query the approved tables listed above.",
        "2. Only use columns defined in the schema. NEVER invent columns.",
        "3. Only generate a single SELECT statement. Never use DDL, DML, PRAGMA, ATTACH, or multi-statement queries.",
        "4. All multi-table JOINs MUST join on sk_id_curr.",
        "5. Always alias joined tables (e.g. FROM applicants a JOIN bureau_summary b ON a.sk_id_curr = b.sk_id_curr).",
        "6. Apply LIMIT 100 to result sets unless the query is a natural aggregate returning 1 row.",
        "7. Use meaningful business aliases for aggregated expressions.",
        "8. For 'top N' requests, use ORDER BY ... DESC LIMIT N.",
    ]
    return "\n".join(lines)
