"""
Talk-to-Data Database Management & Query Execution Module
==========================================================
Provides SQLite database initialization, data population (from local source or
reproducible representative sample), and secure read-only SQL query execution.
"""

import os
import re
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.talk_to_data.schema import TABLE_NAME, get_allowed_columns

DEFAULT_DB_PATH = os.path.join("data", "credit_risk_analytics.db")
DEFAULT_SCHEMA_PATH = os.path.join("sql", "schema.sql")

# Forbidden SQL keywords for read-only safety guardrail
FORBIDDEN_KEYWORDS = {
    "INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE",
    "REPLACE", "CREATE", "ATTACH", "DETACH", "PRAGMA", "VACUUM", "EXEC"
}


def is_read_only_query(sql_query: str) -> bool:
    """
    Validate that an SQL statement is strictly a read-only SELECT query.
    """
    cleaned = re.sub(r"--.*$", "", sql_query, flags=re.MULTILINE)  # strip line comments
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)    # strip block comments
    cleaned = cleaned.strip()

    if not cleaned:
        return False

    # Must start with SELECT or WITH (for CTEs)
    first_word = cleaned.split()[0].upper()
    if first_word not in {"SELECT", "WITH"}:
        return False

    # Check for forbidden mutation keywords as whole tokens
    tokens = set(re.findall(r"\b[A-Za-z_]+\b", cleaned.upper()))
    if any(keyword in tokens for keyword in FORBIDDEN_KEYWORDS):
        return False

    return True


def init_database(
    db_path: Optional[str] = None,
    schema_path: Optional[str] = None
) -> None:
    """
    Initialize SQLite database tables and indexes using sql/schema.sql.
    """
    target_db = db_path or DEFAULT_DB_PATH
    target_schema = schema_path or DEFAULT_SCHEMA_PATH

    os.makedirs(os.path.dirname(target_db), exist_ok=True)

    if not os.path.exists(target_schema):
        raise FileNotFoundError(f"Schema DDL file not found at: {target_schema}")

    with open(target_schema, "r", encoding="utf-8") as f:
        ddl_script = f.read()

    with sqlite3.connect(target_db) as conn:
        conn.executescript(ddl_script)
        conn.commit()


def populate_from_csv(
    csv_path: str,
    db_path: Optional[str] = None,
    limit: Optional[int] = None
) -> int:
    """
    Populate analytics table from an existing processed Home Credit CSV.
    """
    target_db = db_path or DEFAULT_DB_PATH
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Source CSV not found at: {csv_path}")

    df = pd.read_csv(csv_path, nrows=limit)
    allowed_cols = get_allowed_columns()

    # Align column names to lowercase
    df.columns = [c.lower() for c in df.columns]
    present_cols = [c for c in allowed_cols if c in df.columns]
    df_to_insert = df[present_cols]

    with sqlite3.connect(target_db) as conn:
        df_to_insert.to_sql(TABLE_NAME, conn, if_exists="append", index=False)
        conn.commit()

    return len(df_to_insert)


def populate_representative_sample(
    db_path: Optional[str] = None,
    n_samples: int = 1000,
    random_state: int = 42
) -> int:
    """
    Generate a statistically representative sample dataset reflecting the Home Credit
    distributions (e.g. ~8% default rate, realistic income, loan leverage, education groups)
    without requiring large raw CSV downloads.
    """
    target_db = db_path or DEFAULT_DB_PATH
    np.random.seed(random_state)

    sk_ids = np.arange(100001, 100001 + n_samples)
    
    # Target distribution: ~8.0% defaults, matching Home Credit benchmark
    target = np.random.choice([0, 1], size=n_samples, p=[0.92, 0.08])

    contract_types = np.random.choice(["Cash loans", "Revolving loans"], size=n_samples, p=[0.90, 0.10])
    genders = np.random.choice(["F", "M"], size=n_samples, p=[0.66, 0.34])
    car_ownership = np.random.choice(["N", "Y"], size=n_samples, p=[0.66, 0.34])
    realty_ownership = np.random.choice(["Y", "N"], size=n_samples, p=[0.69, 0.31])
    children = np.random.choice([0, 1, 2, 3], size=n_samples, p=[0.70, 0.20, 0.08, 0.02])

    # Income distributions: log-normal with realistic bank values
    incomes = np.round(np.random.lognormal(mean=11.9, sigma=0.45, size=n_samples) / 1000) * 1000
    incomes = np.clip(incomes, 45000, 1500000)

    # Loan amounts roughly 2x to 5x income
    credit_multipliers = np.random.uniform(1.5, 5.0, size=n_samples)
    credits = np.round(incomes * credit_multipliers / 500) * 500
    annuities = np.round(credits / np.random.uniform(12, 28, size=n_samples) / 100) * 100
    goods_prices = np.round(credits * np.random.uniform(0.85, 1.0, size=n_samples) / 500) * 500

    education_choices = [
        "Secondary / secondary special",
        "Higher education",
        "Incomplete higher",
        "Lower secondary"
    ]
    educations = np.random.choice(education_choices, size=n_samples, p=[0.71, 0.24, 0.04, 0.01])

    income_types = np.random.choice(
        ["Working", "Commercial associate", "Pensioner", "State servant"],
        size=n_samples,
        p=[0.52, 0.23, 0.18, 0.07]
    )

    family_statuses = np.random.choice(
        ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        size=n_samples,
        p=[0.64, 0.15, 0.10, 0.06, 0.05]
    )

    housing_types = np.random.choice(
        ["House / apartment", "With parents", "Municipal apartment", "Rented apartment"],
        size=n_samples,
        p=[0.88, 0.05, 0.04, 0.03]
    )

    ages = np.round(np.random.uniform(21, 68, size=n_samples), 1)
    employments = np.round(np.clip(np.random.uniform(0.2, 35, size=n_samples), 0, ages - 18), 1)

    occupations = np.random.choice(
        ["Laborers", "Core staff", "Managers", "Drivers", "Sales staff", "High skill tech staff", "Accountants"],
        size=n_samples,
        p=[0.32, 0.18, 0.14, 0.12, 0.12, 0.06, 0.06]
    )

    organizations = np.random.choice(
        ["Business Entity Type 3", "Self-employed", "Other", "Government", "School", "Trade: type 3", "Transport: type 4"],
        size=n_samples,
        p=[0.35, 0.18, 0.15, 0.12, 0.08, 0.06, 0.06]
    )

    # External credit scores (correlated with target)
    ext_1 = np.where(target == 1, np.random.beta(2, 5, size=n_samples), np.random.beta(5, 3, size=n_samples))
    ext_2 = np.where(target == 1, np.random.beta(2, 4, size=n_samples), np.random.beta(4, 2, size=n_samples))
    ext_3 = np.where(target == 1, np.random.beta(2, 5, size=n_samples), np.random.beta(4, 3, size=n_samples))

    # Derived ratios
    credit_income_ratio = np.round(credits / incomes, 2)
    annuity_income_ratio = np.round(annuities / incomes, 2)
    credit_goods_ratio = np.round(credits / goods_prices, 2)

    # Historical aggregations
    bureau_counts = np.random.poisson(lam=4, size=n_samples)
    bureau_debts = np.round(bureau_counts * np.random.uniform(20000, 150000, size=n_samples), 2)
    prev_counts = np.random.poisson(lam=3, size=n_samples)
    prev_refusals = np.where(prev_counts > 0, np.random.choice([0.0, 0.25, 0.5, 1.0], size=n_samples, p=[0.70, 0.15, 0.10, 0.05]), 0.0)
    total_payments = np.round(credits * np.random.uniform(0.2, 1.5, size=n_samples), 2)
    late_ratios = np.where(target == 1, np.random.uniform(0.1, 0.6, size=n_samples), np.random.uniform(0.0, 0.1, size=n_samples))

    data_dict = {
        "sk_id_curr": sk_ids,
        "target": target,
        "name_contract_type": contract_types,
        "code_gender": genders,
        "flag_own_car": car_ownership,
        "flag_own_realty": realty_ownership,
        "cnt_children": children,
        "amt_income_total": incomes,
        "amt_credit": credits,
        "amt_annuity": annuities,
        "amt_goods_price": goods_prices,
        "name_income_type": income_types,
        "name_education_type": educations,
        "name_family_status": family_statuses,
        "name_housing_type": housing_types,
        "age_years": ages,
        "employment_years": employments,
        "occupation_type": occupations,
        "organization_type": organizations,
        "ext_source_1": np.round(ext_1, 4),
        "ext_source_2": np.round(ext_2, 4),
        "ext_source_3": np.round(ext_3, 4),
        "credit_income_ratio": credit_income_ratio,
        "annuity_income_ratio": annuity_income_ratio,
        "credit_goods_ratio": credit_goods_ratio,
        "bureau_credit_count": bureau_counts,
        "bureau_total_debt": bureau_debts,
        "prev_app_count": prev_counts,
        "prev_refusal_rate": prev_refusals,
        "total_payment_amount": total_payments,
        "late_payment_ratio": np.round(late_ratios, 4)
    }

    df_sample = pd.DataFrame(data_dict)

    with sqlite3.connect(target_db) as conn:
        df_sample.to_sql(TABLE_NAME, conn, if_exists="append", index=False)

        # Populate relational tables
        app_cols = get_allowed_columns("applicants")
        df_sample[app_cols].to_sql("applicants", conn, if_exists="append", index=False)
        df_sample[["sk_id_curr", "bureau_credit_count", "bureau_total_debt"]].to_sql(
            "bureau_summary", conn, if_exists="append", index=False
        )
        df_sample[["sk_id_curr", "prev_app_count", "prev_refusal_rate"]].to_sql(
            "previous_application_summary", conn, if_exists="append", index=False
        )
        df_sample[["sk_id_curr", "total_payment_amount", "late_payment_ratio"]].to_sql(
            "installment_summary", conn, if_exists="append", index=False
        )
        conn.commit()

    return len(df_sample)


def execute_query(
    sql_query: str,
    db_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Safely execute a read-only SQL query against the analytics database.

    Args:
        sql_query: SQL SELECT query string.
        db_path: Database path (defaults to data/credit_risk_analytics.db).

    Returns:
        pandas DataFrame containing the query results.

    Raises:
        PermissionError: If query contains mutation statements.
        sqlite3.Error: If SQL execution fails.
    """
    target_db = db_path or DEFAULT_DB_PATH
    if not os.path.exists(target_db):
        raise FileNotFoundError(
            f"Database file not found at {target_db}. Run init_database() first."
        )

    if not is_read_only_query(sql_query):
        raise PermissionError(
            "Query rejected: Only read-only SELECT queries are permitted."
        )

    # Connect with query_only pragma for defense in depth
    conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(sql_query, conn)
        return df
    finally:
        conn.close()


def execute_query_safe(
    sql_query: str,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Non-throwing, structured wrapper for query execution.

    Returns:
        Dict with keys: success, rows, columns, row_count, error
    """
    try:
        df = execute_query(sql_query, db_path)
        return {
            "success": True,
            "rows": df.to_dict(orient="records"),
            "columns": list(df.columns),
            "row_count": len(df),
            "error": None
        }
    except Exception as e:
        logger.error("Query execution failed: %s", e, exc_info=True)
        return {
            "success": False,
            "rows": [],
            "columns": [],
            "row_count": 0,
            "error": "A database error occurred while executing the query."
        }


def get_database_summary(db_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Return basic statistical summary of the analytics database.
    """
    target_db = db_path or DEFAULT_DB_PATH
    if not os.path.exists(target_db):
        return {"exists": False}

    summary_query = f"""
    SELECT 
        COUNT(*) AS row_count,
        ROUND(AVG(target) * 100, 2) AS default_rate_pct,
        ROUND(AVG(amt_income_total), 2) AS avg_income,
        ROUND(AVG(amt_credit), 2) AS avg_credit
    FROM {TABLE_NAME};
    """
    df = execute_query(summary_query, target_db)
    return {
        "exists": True,
        "table": TABLE_NAME,
        "row_count": int(df["row_count"].iloc[0]),
        "default_rate_pct": float(df["default_rate_pct"].iloc[0]),
        "avg_income": float(df["avg_income"].iloc[0]),
        "avg_credit": float(df["avg_credit"].iloc[0])
    }
