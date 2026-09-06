-- ==============================================================================
-- Credit Risk Analytics Database Schema (SQLite)
-- Database: data/credit_risk_analytics.db
-- Purpose: Analytical querying for Talk-to-Data conversational analytics.
-- ==============================================================================

-- 1. Unified Analytics Table (Preserved for single-table queries and backward compatibility)
CREATE TABLE IF NOT EXISTS applicant_analytics (
    sk_id_curr                  INTEGER PRIMARY KEY,
    target                      INTEGER NOT NULL CHECK(target IN (0, 1)),
    name_contract_type          TEXT NOT NULL,
    code_gender                 TEXT,
    flag_own_car                TEXT,
    flag_own_realty             TEXT,
    cnt_children                INTEGER DEFAULT 0,
    amt_income_total            REAL NOT NULL,
    amt_credit                  REAL NOT NULL,
    amt_annuity                 REAL,
    amt_goods_price             REAL,
    name_income_type            TEXT,
    name_education_type         TEXT,
    name_family_status          TEXT,
    name_housing_type           TEXT,
    age_years                   REAL,
    employment_years            REAL,
    occupation_type             TEXT,
    organization_type           TEXT,
    ext_source_1                REAL,
    ext_source_2                REAL,
    ext_source_3                REAL,
    credit_income_ratio         REAL,
    annuity_income_ratio        REAL,
    credit_goods_ratio          REAL,
    bureau_credit_count         INTEGER DEFAULT 0,
    bureau_total_debt           REAL DEFAULT 0.0,
    prev_app_count              INTEGER DEFAULT 0,
    prev_refusal_rate           REAL DEFAULT 0.0,
    total_payment_amount        REAL DEFAULT 0.0,
    late_payment_ratio          REAL DEFAULT 0.0
);

-- 2. Core Applicants Table (Application-level features)
CREATE TABLE IF NOT EXISTS applicants (
    sk_id_curr                  INTEGER PRIMARY KEY,
    target                      INTEGER NOT NULL CHECK(target IN (0, 1)),
    name_contract_type          TEXT NOT NULL,
    code_gender                 TEXT,
    flag_own_car                TEXT,
    flag_own_realty             TEXT,
    cnt_children                INTEGER DEFAULT 0,
    amt_income_total            REAL NOT NULL,
    amt_credit                  REAL NOT NULL,
    amt_annuity                 REAL,
    amt_goods_price             REAL,
    name_income_type            TEXT,
    name_education_type         TEXT,
    name_family_status          TEXT,
    name_housing_type           TEXT,
    age_years                   REAL,
    employment_years            REAL,
    occupation_type             TEXT,
    organization_type           TEXT,
    ext_source_1                REAL,
    ext_source_2                REAL,
    ext_source_3                REAL,
    credit_income_ratio         REAL,
    annuity_income_ratio        REAL,
    credit_goods_ratio          REAL
);

-- 3. Bureau Historical Summary Table
CREATE TABLE IF NOT EXISTS bureau_summary (
    sk_id_curr                  INTEGER PRIMARY KEY,
    bureau_credit_count         INTEGER DEFAULT 0,
    bureau_total_debt           REAL DEFAULT 0.0,
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr)
);

-- 4. Previous Applications Summary Table
CREATE TABLE IF NOT EXISTS previous_application_summary (
    sk_id_curr                  INTEGER PRIMARY KEY,
    prev_app_count              INTEGER DEFAULT 0,
    prev_refusal_rate           REAL DEFAULT 0.0,
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr)
);

-- 5. Installment Payments Summary Table
CREATE TABLE IF NOT EXISTS installment_summary (
    sk_id_curr                  INTEGER PRIMARY KEY,
    total_payment_amount        REAL DEFAULT 0.0,
    late_payment_ratio          REAL DEFAULT 0.0,
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr)
);

-- Analytical Indexes for Fast Query Execution
CREATE INDEX IF NOT EXISTS idx_applicant_target ON applicant_analytics(target);
CREATE INDEX IF NOT EXISTS idx_applicant_contract_type ON applicant_analytics(name_contract_type);
CREATE INDEX IF NOT EXISTS idx_applicant_education ON applicant_analytics(name_education_type);
CREATE INDEX IF NOT EXISTS idx_applicant_income_type ON applicant_analytics(name_income_type);
CREATE INDEX IF NOT EXISTS idx_applicant_organization ON applicant_analytics(organization_type);
CREATE INDEX IF NOT EXISTS idx_applicant_income ON applicant_analytics(amt_income_total);
CREATE INDEX IF NOT EXISTS idx_applicant_credit ON applicant_analytics(amt_credit);

CREATE INDEX IF NOT EXISTS idx_applicants_target ON applicants(target);
CREATE INDEX IF NOT EXISTS idx_applicants_education ON applicants(name_education_type);
CREATE INDEX IF NOT EXISTS idx_applicants_income ON applicants(amt_income_total);
CREATE INDEX IF NOT EXISTS idx_applicants_credit ON applicants(amt_credit);

CREATE INDEX IF NOT EXISTS idx_bureau_debt ON bureau_summary(bureau_total_debt);
CREATE INDEX IF NOT EXISTS idx_bureau_count ON bureau_summary(bureau_credit_count);

CREATE INDEX IF NOT EXISTS idx_prev_refusal ON previous_application_summary(prev_refusal_rate);
CREATE INDEX IF NOT EXISTS idx_prev_count ON previous_application_summary(prev_app_count);

CREATE INDEX IF NOT EXISTS idx_inst_late_ratio ON installment_summary(late_payment_ratio);
CREATE INDEX IF NOT EXISTS idx_inst_payment ON installment_summary(total_payment_amount);

-- 6. Model-Predicted Risk Scores Table
-- IMPORTANT: Values are derived from the trained CatBoost model (predict_proba).
-- Do NOT use the 'target' column as a substitute for predicted risk.
CREATE TABLE IF NOT EXISTS applicant_risk_scores (
    sk_id_curr          INTEGER PRIMARY KEY,
    default_probability REAL NOT NULL,          -- Raw CatBoost predicted probability of default (0.0-1.0)
    risk_score          REAL NOT NULL,          -- Scaled 0-100 risk score (default_probability * 100)
    risk_band           TEXT NOT NULL,          -- Risk band: 'Low', 'Medium', or 'High'
    FOREIGN KEY (sk_id_curr) REFERENCES applicants(sk_id_curr)
);

CREATE INDEX IF NOT EXISTS idx_risk_band ON applicant_risk_scores(risk_band);
CREATE INDEX IF NOT EXISTS idx_default_probability ON applicant_risk_scores(default_probability);
CREATE INDEX IF NOT EXISTS idx_risk_score ON applicant_risk_scores(risk_score);
