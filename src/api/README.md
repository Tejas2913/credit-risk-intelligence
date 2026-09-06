# Credit Risk Intelligence Platform — Unified FastAPI Backend

The `src/api` module provides an enterprise-grade, unified REST API exposing the three core pillars of the Credit Risk Intelligence Platform:
1. **Module 1: Portfolio EDA & Business Insights** (`/api/eda/*`)
2. **Module 2: ML Credit Risk Inference & Explainability (SHAP/XAI)** (`/api/ml/*`)
3. **Module 3: Talk-to-Data NL-to-SQL Analytics Assistant** (`/api/chat`)

---

## 1. System Architecture

```
                                  Client Requests
                                         │
                                         ▼
                      ┌──────────────────────────────────────┐
                      │        FastAPI Unified Backend       │
                      │         (src.api.main:app)           │
                      └──────────────────┬───────────────────┘
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
        ▼                                ▼                                ▼
┌───────────────┐              ┌───────────────────┐             ┌───────────────────┐
│ Module 1: EDA │              │ Module 2: ML & XAI│             │ Module 3: Chatbot │
│ (src/api/eda) │              │  (src/api/ml)     │             │ (src/api/chat)    │
└───────┬───────┘              └─────────┬─────────┘             └─────────┬─────────┘
        │                                │                                 │
        │ Direct SQL                     │ Feature Vector                  │ Schema-Aware SQL
        ▼                                ▼                                 ▼
┌─────────────────────────┐    ┌───────────────────┐             ┌───────────────────┐
│ credit_risk_analytics.db│    │ model_features.db │             │   LLM Engine /    │
│  (109.55 MB, Read-Only) │    │(272.67MB,ReadOnly)│             │ Deterministic     │
└─────────────────────────┘    └─────────┬─────────┘             └─────────┬─────────┘
                                         │                                 │
                                         ▼                                 ▼
                               ┌───────────────────┐             ┌───────────────────┐
                               │  CatBoost Model   │             │   SQL Validator   │
                               │  + TreeExplainer  │             │  (Read-Only DB)   │
                               └───────────────────┘             └───────────────────┘
```

---

## 2. API Endpoints Reference

### Health & Root Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health check reporting SQLite database status and CatBoost model readiness. |
| `GET` | `/` | Service root and navigation links. |
| `GET` | `/docs` | Interactive Swagger / OpenAPI documentation UI. |
| `GET` | `/redoc` | ReDoc API specification. |

---

### Module 1: EDA & Business Insights (`/api/eda`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/eda/summary` | Returns total applicants (307,511), default counts (24,825), non-default counts (282,686), baseline default rate (8.07%), and feature counts. |
| `GET` | `/api/eda/insights` | Returns all 6 authoritative business insights with chart labels, values, sample counts, and business interpretations. |

#### Example: `GET /api/eda/summary`
```json
{
  "total_applicants": 307511,
  "default_applicants": 24825,
  "non_default_applicants": 282686,
  "default_rate": 0.080729,
  "default_rate_pct": 8.07,
  "total_features": 231,
  "numerical_features": 106,
  "categorical_features": 16
}
```

---

### Module 2: ML Inference & SHAP Explainability (`/api/ml`)

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/ml/applicant/{applicant_id}` | Retrieves 231 features from `model_features.db` and computes calibrated default probability, risk score (0–100), risk band, and decision-support recommendation. |
| `GET` | `/api/ml/applicant/{applicant_id}/explanation?top_k=5` | Computes TreeSHAP feature attribution values, ranking top risk-increasing and risk-reducing drivers. Protected demographic attributes are excluded from explanations. |

#### Example: `GET /api/ml/applicant/396899`
```json
{
  "success": true,
  "applicant_id": 396899,
  "default_probability": 0.228251,
  "risk_score": 22.83,
  "risk_band": "Low",
  "prediction": 0,
  "decision": "Approve / Low Risk"
}
```

#### Example: `GET /api/ml/applicant/396899/explanation?top_k=3`
```json
{
  "success": true,
  "applicant_id": 396899,
  "base_value": -0.022849,
  "risk_increasing_factors": [
    {
      "feature": "EXT_SOURCE_1",
      "feature_value": "missing",
      "shap_value": 0.124376,
      "direction": "Increases Risk",
      "description": "External credit bureau score 1 (normalized)"
    }
  ],
  "risk_reducing_factors": [
    {
      "feature": "EXT_SOURCE_2",
      "feature_value": 0.594,
      "shap_value": -0.182214,
      "direction": "Reduces Risk",
      "description": "External credit bureau score 2 (normalized)"
    }
  ],
  "disclaimer": "Decision-support explanation only. Excludes protected demographic attributes (gender, family status, education, assets) in compliance with fair-lending standards."
}
```

---

### Module 3: Talk-to-Data Chatbot (`/api/chat`)

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Accepts natural language questions about portfolio analytics, generates safe SQL queries, executes them against `credit_risk_analytics.db`, and returns synthesized executive answers. |

#### Example Request: `POST /api/chat`
```json
{
  "message": "What is the default rate across education levels?",
  "session_id": "session_001"
}
```

#### Example Response:
```json
{
  "success": true,
  "answer": "Analysis across education levels shows the highest default risk in 'Lower secondary' at 10.93% (3,816 applicants), while 'Academic degree' exhibits the lowest default risk at 1.83% (164 applicants).",
  "sql": "SELECT name_education_type, COUNT(*) AS applicant_count, ROUND(AVG(target) * 100, 2) AS default_rate_pct, ROUND(AVG(amt_credit), 2) AS avg_credit_amount, ROUND(AVG(amt_income_total), 2) AS avg_income FROM applicant_analytics GROUP BY name_education_type ORDER BY default_rate_pct DESC;",
  "data": [
    {"name_education_type": "Lower secondary", "applicant_count": 3816, "default_rate_pct": 10.93, "avg_credit_amount": 491699.0, "avg_income": 128866.0},
    {"name_education_type": "Secondary / secondary special", "applicant_count": 218391, "default_rate_pct": 8.94, "avg_credit_amount": 571644.0, "avg_income": 155108.0},
    {"name_education_type": "Incomplete higher", "applicant_count": 10277, "default_rate_pct": 8.48, "avg_credit_amount": 565922.0, "avg_income": 181561.0},
    {"name_education_type": "Higher education", "applicant_count": 74863, "default_rate_pct": 5.36, "avg_credit_amount": 689809.0, "avg_income": 208603.0},
    {"name_education_type": "Academic degree", "applicant_count": 164, "default_rate_pct": 1.83, "avg_credit_amount": 729506.0, "avg_income": 240000.0}
  ],
  "columns": ["name_education_type", "applicant_count", "default_rate_pct", "avg_credit_amount", "avg_income"],
  "row_count": 5,
  "intent": "education_risk_analysis",
  "used_llm": false,
  "error": null
}
```

---

## 3. How to Run the API Server

Start the development server with live reloading:

```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Access the interactive OpenAPI Documentation:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 4. Runtime Independence & Data Integrity

- **Zero Raw CSV Dependencies**: The runtime backend relies entirely on `data/credit_risk_analytics.db` (109.55 MB) and `data/model_features.db` (272.67 MB). The original 2.64 GB raw CSV dataset is not required at runtime.
- **Strict Read-Only Access**: All SQLite connections are opened using URI mode `?mode=ro`.
- **SQL Security Guardrails**: The multi-layer SQL validator blocks all DDL/DML commands (`DROP`, `DELETE`, `UPDATE`, `INSERT`), semicolons, comments, and internal SQLite tables.
- **Fair Lending Compliance**: Sensitive demographic attributes (`CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`) are automatically excluded from user-facing feature explanations.
- **Decision-Support Guardrail**: Individual underwriting questions submitted to the Talk-to-Data chatbot are intercepted and redirected to the ML inference engine.
