# AI-Powered Credit Risk Intelligence Platform

An end-to-end AI-powered credit risk decision-support platform built on the Home Credit Default Risk dataset.

The platform provides three core capabilities exposed through a unified FastAPI backend and a modern React + Tailwind CSS dashboard:
1. **Module 1: Exploratory Data Analysis & Business Insights** (`/api/eda/*`)
2. **Module 2: ML Credit Risk Inference & Explainability (TreeSHAP XAI)** (`/api/ml/*`)
3. **Module 3: Talk-to-Data Natural Language to SQL Analytics Assistant** (`/api/chat`)

---

## 1. System Architecture

### Component Architecture Diagram

```
                                  User / Web Browser
                                          │
                                          ▼
                      ┌───────────────────────────────────────┐
                      │       Frontend Container (Nginx)      │
                      │       React 18 + Vite (Port 3000)     │
                      └───────────────────┬───────────────────┘
                                          │
                                          │ Internal Reverse Proxy (/api/*, /health)
                                          ▼
                      ┌───────────────────────────────────────┐
                      │       Backend Container (Uvicorn)     │
                      │         FastAPI (Port 8000)           │
                      └───┬───────────────────┬─────────────┬─┘
                          │                   │             │
        ┌─────────────────┘                   │             └─────────────────┐
        ▼                                     ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
│     EDA & Insights API    │   │    ML Risk Analysis API   │   │     Talk-to-Data API      │
│  - 6 Key Business Charts  │   │  - Feature Store Lookup   │   │  - Safety/Capability Check│
│  - Dataset Distribution   │   │    (model_features.db)    │   │  - Fast Deterministic Path│
│  - Portfolio KPI Metrics  │   │  - CatBoost Model Scoring │   │  - Schema-Aware LLM Path  │
│  - Read-Only Analytics DB │   │  - TreeSHAP Explainer     │   │    (Groq / OpenRouter)    │
│                           │   │  - Demographic Masking    │   │  - Multi-Layer Validator  │
│                           │   │  - Decision-Support Rules │   │  - Read-Only SQLite Engine│
│                           │   │                           │   │  - Grounded Business Text │
└─────────────┬─────────────┘   └─────────────┬─────────────┘   └─────────────┬─────────────┘
              │                               │                               │
              ▼                               ▼                               ▼
┌───────────────────────────┐   ┌───────────────────────────┐   ┌───────────────────────────┐
│ credit_risk_analytics.db  │   │     model_features.db     │   │ credit_risk_analytics.db  │
│   (307,511 Applicants)    │   │   (231 Features/App ID)   │   │  - applicants             │
│   - applicant_analytics   │   │   - model_features        │   │  - bureau_summary         │
│   - applicant_risk_scores │   │                           │   │  - previous_app_summary   │
│   - relational summaries  │   │                           │   │  - installment_summary    │
│     (Read-Only, ?mode=ro) │   │     (Read-Only, ?mode=ro) │   │  - applicant_risk_scores  │
└───────────────────────────┘   └───────────────────────────┘   └───────────────────────────┘
```

### Docker Network & Runtime Flow
- **Browser Client**: Interacts with the frontend at `http://localhost:3000`.
- **Nginx Reverse Proxy**: Serves compiled React SPA static assets and proxies API calls (`/api/*`, `/health`) to the FastAPI backend over the internal Docker network `credit-risk-net`.
- **FastAPI Backend**: Runs inside its container listening on internal port `8000` (not exposed directly on the host for security). It loads precomputed ML artifacts and queries pre-built SQLite databases in read-only mode (`?mode=ro`).
- **Runtime Databases vs. Raw Data**: The original raw Home Credit CSV files (~2.64 GB) were used during offline feature engineering and database preparation. They are **not required at runtime** and are excluded from Docker packaging (`.dockerignore`). Only the processed SQLite databases (`data/credit_risk_analytics.db` and `data/model_features.db`) and trained model artifacts (`artifacts/`) are used at runtime.

---

## 2. Key Features

- **Portfolio-Level Business Insights**: Interactive EDA charts highlighting historical associations between loan default rates and previous loan refusals, POS/CASH delinquency, credit card utilization, installment payment completeness, and late-payment frequency.
- **Applicant Risk Scoring**: Instant default probability prediction (0–100 standardized risk score) and risk band categorization (`Low`, `Medium`, `High`) with decision-support recommendations.
- **Explainable AI (TreeSHAP)**: Global feature importance rankings and local per-applicant waterfall explanations showing top risk-increasing and risk-mitigating feature contributions.
- **Fair Lending Masking**: User-facing SHAP visualizer automatically excludes protected demographic attributes (`CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`) from end-user explanations.
- **General-Purpose Talk-to-Data Assistant**: Schema-grounded conversational analytics answering any reasonable analytical question answerable from the approved database schema, powered by Groq (`openai/gpt-oss-120b`) or OpenRouter (`meta-llama/llama-3.3-70b-instruct`) with deterministic fast-path shortcuts for common questions.
- **Model-Derived Risk Table (`applicant_risk_scores`)**: Integrated portfolio table scoring all 307,511 applicants with CatBoost predictions, enabling risk-tier aggregations and filtering in natural language.
- **Multi-Layer SQL Safety**: Strict read-only database connections, AST/keyword parsing, single-statement enforcement, SQLite system table blocking, approved relational join validation, and individual underwriting redirection.
- **100% Offline Deterministic Fallback**: Automatic fallback to regex-based SQL templates when no LLM API key is provided or when external APIs are unreachable.

---

## 3. Technology Stack

| Layer | Technologies |
|---|---|
| **Backend Framework** | Python 3.10+, FastAPI, Uvicorn, Pydantic v2 |
| **Machine Learning & XAI** | CatBoost 1.2+, LightGBM, XGBoost, Scikit-learn, SHAP (TreeExplainer) |
| **Data Storage & Querying** | SQLite3 (Read-Only mode `file:...?mode=ro`), Pandas, NumPy |
| **LLM & NL-to-SQL** | Groq API (`openai/gpt-oss-120b`), OpenRouter API (`meta-llama/llama-3.3-70b-instruct`) |
| **Frontend UI** | React 18, Vite, Tailwind CSS, Lucide React, Recharts |
| **Containerization & Web Server** | Docker, Docker Compose, Nginx (Alpine-based production build) |

---

## 4. Dataset and Data Processing

- **Source Dataset**: Home Credit Default Risk dataset (Kaggle).
- **Total Applicants**: 307,511 unique loan applications (`SK_ID_CURR`).
- **Target Variable (`TARGET`)**:
  - `0`: Loan repaid on time (282,686 applicants — **91.93%**)
  - `1`: Client experienced payment difficulties / default (24,825 applicants — **8.07%**)
- **Data Stores**:
  1. `data/credit_risk_analytics.db` (261.46 MB): Contains normalized analytical tables (`applicants`, `bureau_summary`, `previous_application_summary`, `installment_summary`, `applicant_risk_scores`, and legacy `applicant_analytics`) indexed on `SK_ID_CURR` for instant EDA and relational Talk-to-Data queries.
  2. `data/model_features.db` (285.91 MB): Compact SQLite inference feature store containing all 231 pre-engineered features per applicant, enabling sub-millisecond feature retrieval for real-time model scoring without raw CSV merges.

---

## 5. EDA and Business Insights

Analysis of the 307,511 historical loan applications revealed strong observational relationships between past financial behavior and loan default rates:

1. **Overall Class Imbalance**: Baseline default rate across the historical portfolio is **8.07%** (approximately 11.4 non-defaults for every 1 default).
2. **Historical Application Refusals**: Applicants with previously refused loans exhibited an observed default rate of **12.0%**, compared to **7.6%** for applicants whose previous loans were approved.
3. **POS / Cash Loan Delinquency**: Applicants with high Days Past Due (DPD) on historical POS/Cash loans were observed to default at **12.0%**, versus **7.7%** for those with zero DPD.
4. **Credit Card Utilization**: Borrowers in the highest credit card balance utilization band had an observed default rate of **19.5%**, compared to **5.4%** for zero-utilization borrowers.
5. **Installment Payment Behavior**: Borrowers with low historical installment payment completion ratios defaulted at **13.5%**, compared to **7.5%** for borrowers with full installment completion.
6. **Late-Payment Frequency**: Borrowers with frequent late installment payments defaulted at **12.0%**, compared to **6.8%** for borrowers with clean payment histories.

*(Note: These findings represent observational associations across historical data, not direct causal relationships.)*

---

## 6. Model Selection Rationale and Class Imbalance Strategy

### Model Selection Evaluation
Four machine learning algorithms were trained and evaluated on a stratified validation split (80% train / 20% validation, 61,503 validation applicants):

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 Score |
|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.7735 | 0.2608 | 0.1749 | 0.6975 | 0.2797 |
| **XGBoost** | 0.7874 | 0.2848 | 0.2045 | 0.6455 | 0.3107 |
| **LightGBM** | 0.7870 | 0.2844 | 0.1930 | 0.6806 | 0.3008 |
| **CatBoost (Selected Final Model)** | **0.7878** | **0.2862** | 0.2009 | 0.6618 | 0.3082 |

### Selection Rationale
- **Primary Metric (PR-AUC)**: Due to severe class imbalance (8.07% default rate), Area Under the Precision-Recall Curve (PR-AUC) was chosen as the primary selection criterion because ROC-AUC can present an overly optimistic view when the majority class dominates.
- **Performance**: CatBoost achieved the highest PR-AUC (**0.2862**) and highest ROC-AUC (**0.7878**) among all evaluated candidates.
- **Categorical Handling**: CatBoost natively handles categorical variables, preserving categorical structure without high-dimensional sparse one-hot encoding.

### Class Imbalance Handling & Threshold Tuning
- **Stratified Splitting**: Stratified train/validation splitting ensured identical 8.07% default proportions in training and evaluation sets.
- **Class Weighting**: CatBoost was trained with class weights to penalize false negatives proportionally.
- **Threshold Optimization**: The model comparison table above reflects standard evaluation metrics. The final operating classification threshold was subsequently tuned on the validation set to maximize F1 score, yielding an optimal decision threshold of **0.66**.
- *Important*: The 0.66 threshold was selected on the validation set to maximize F1 for this prototype and is not a universal lending policy threshold.

---

## 7. Model Evaluation and Performance Results

### Final CatBoost Model Validation Results (Operating Threshold = 0.66)

| Metric | Score | Business Meaning |
|---|---|---|
| **ROC-AUC** | **0.7878** | Ability of the model to rank risky borrowers above safe borrowers across all possible cutoffs. |
| **PR-AUC** | **0.2862** | Precision-Recall trade-off specifically focusing on the rare positive default class (baseline = 0.0807). |
| **Precision** | **0.2824** | Of all applicants flagged by the model as high risk, 28.24% were actual defaults. |
| **Recall** | **0.4354** | The model successfully identified 43.54% of all actual defaulters in the validation population. |
| **F1 Score** | **0.3426** | Harmonic mean balancing precision and recall at the 0.66 threshold (optimized from 0.3082 baseline). |
| **Accuracy** | **0.8651** | Overall percentage of correct classifications (86.51%). |

### Confusion Matrix (Validation Set: N = 61,503)

| | Predicted Non-Default (0) | Predicted Default (1) | Total Actual |
|---|---|---|---|
| **Actual Non-Default (0)** | **51,044** (TN) | **5,494** (FP) | 56,538 |
| **Actual Default (1)** | **2,803** (FN) | **2,162** (TP) | 4,965 |
| **Total Predicted** | 53,847 | 7,656 | 61,503 |

*(Note: Model comparison metrics evaluate ranking capability across all thresholds, whereas the metrics above reflect operational classification at the tuned 0.66 cutoff.)*

---

## 8. Explainable AI (XAI) & SHAP Attributions

The platform utilizes **TreeSHAP** (`shap.TreeExplainer`) for model interpretability at both portfolio and applicant levels:

### Global Feature Importance
The top 10 features driving global credit risk predictions across the portfolio are:
1. `EXT_SOURCE_2`: Normalized external credit bureau score 2.
2. `EXT_SOURCE_3`: Normalized external credit bureau score 3.
3. `EXT_SOURCE_1`: Normalized external credit bureau score 1.
4. `POS_AVG_FUTURE_INSTALLMENTS`: Average remaining installments on POS/Cash loans.
5. `LATE_PAYMENT_RATIO`: Ratio of historical late installment payments.
6. `ANNUITY_CREDIT_RATIO`: Loan payment annuity relative to total credit amount.
7. `CREDIT_GOODS_RATIO`: Total credit requested relative to goods price.
8. `AMT_ANNUITY`: Monthly loan repayment annuity amount.
9. `PREV_AVG_CREDIT_APPLICATION_RATIO`: Ratio of credit approved to credit applied for in previous loans.
10. `BUREAU_DEBT_CREDIT_RATIO`: Total outstanding bureau debt relative to credit limit.

*(Note: SHAP values describe how features contribute to the model's prediction; they are not causal explanations.)*

### Local Interpretability & Fair Lending Guardrails
- For any evaluated applicant, the platform calculates local SHAP values showing the exact positive (risk-increasing) and negative (risk-reducing) feature contributions.
- **Protected-Attribute Masking**: To adhere to fair lending standards, protected demographic features (`CODE_GENDER`, `NAME_FAMILY_STATUS`, `NAME_EDUCATION_TYPE`, `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`) are excluded from user-facing SHAP waterfall explanations in the API response. Masking filters presentation without altering model calculation or resulting in zeroed factors.

---

## 9. Rule Derivation Logic and Risk Decisions

The platform translates continuous ML outputs into structured, transparent credit risk recommendations for human underwriters:

```
Predicted Probability (0.0 – 1.0)
                │
                ▼
Risk Score = Probability × 100  (0 – 100 Scale)
                │
                ▼
Risk Band Assignment:
  ├── Probability < 0.30        ──►  LOW RISK     (Approve / Low Risk)
  ├── 0.30 ≤ Probability < 0.60 ──►  MEDIUM RISK  (Manual Review)
  └── Probability ≥ 0.60        ──►  HIGH RISK    (High Risk / Reject Recommendation)
```

### Validation Band Distribution & Observed Default Rates

| Risk Band | Probability Range | Validation Count | Population Share | Observed Default Rate | Decision-Support Recommendation |
|---|---|---|---|---|---|
| **Low Risk** | $< 0.30$ | 28,809 | 46.84% | **2.21%** | Approve / Low Risk |
| **Medium Risk** | $0.30 - 0.599$ | 22,152 | 36.02% | **7.78%** | Manual Review |
| **High Risk** | $\ge 0.60$ | 10,542 | 17.14% | **24.69%** | High Risk / Reject Recommendation |

*Monotonic increase in empirical default rates across risk bands (2.21% $\rightarrow$ 7.78% $\rightarrow$ 24.69%) confirms strong risk separation.*

### Sample Outputs

#### Example 1: Low-Risk Applicant (ID `396899`)
- **Predicted Default Probability**: `0.228251` (22.83%)
- **Risk Score**: `22.83 / 100`
- **Assigned Risk Band**: `Low`
- **Decision-Support Recommendation**: `Approve / Low Risk`
- **Key SHAP Drivers**: Strong external credit scores (`EXT_SOURCE_2` SHAP `-0.1822`), average credit limit (`BUREAU_AVG_LIMIT` SHAP `-0.1546`), and zero historical late payments.

#### Example 2: High-Risk Applicant (ID `100002`)
- **Predicted Default Probability**: `0.827562` (82.76%)
- **Risk Score**: `82.76 / 100`
- **Assigned Risk Band**: `High`
- **Decision-Support Recommendation**: `High Risk / Reject Recommendation`
- **Key SHAP Drivers**: Low external scores (`EXT_SOURCE_3` SHAP `+0.5442`, `EXT_SOURCE_1` SHAP `+0.3771`, `EXT_SOURCE_2` SHAP `+0.2608`), elevated social circle defaults (`DEF_30_CNT_SOCIAL_CIRCLE` SHAP `+0.1246`).

> [!IMPORTANT]
> **Decision-Support Notice**: This platform is an AI-assisted decision-support prototype and does **not** make autonomous lending decisions. All risk scores, risk bands, SHAP feature attributions, and recommendations are designed exclusively to assist qualified human underwriters.

---

## 10. Model-Risk Analytics (`applicant_risk_scores`)

To bridge individual ML scoring with conversational portfolio analytics, the analytical database incorporates a dedicated model-derived table:

### `applicant_risk_scores` Table Schema
- `sk_id_curr` (INTEGER, Primary Key): Unique applicant identifier.
- `default_probability` (REAL): CatBoost model predicted default probability (0.0 to 1.0).
- `risk_score` (REAL): Standardized risk score (`default_probability * 100`).
- `risk_band` (TEXT): Calibrated risk category (`'Low'`, `'Medium'`, `'High'`).

All 307,511 applicants have precomputed CatBoost risk scores matching the live ML predictor.

### Important Distinction: TARGET vs. Predicted Risk
- **`TARGET`**: Historical observed loan outcome from the 2018 dataset (`0` = Repaid, `1` = Defaulted). Used for historical retrospective analytics.
- **`default_probability` / `risk_score`**: Forward-looking CatBoost machine learning model prediction. Used for risk assessment and predictive tiering.
- The platform enforces this distinction so historical default rates and model-predicted risks are never conflated.

### Sample Model-Risk Queries Supported by Talk-to-Data
- *"How many applicants are high risk?"*
- *"What percentage of applicants are classified as high risk?"*
- *"Show me the top 5 high-risk applicants."*
- *"Show the top 10 applicants by predicted risk."*
- *"How many high-risk applicants had previous loan refusals?"*

---

## 11. Talk-to-Data Conversational Assistant & General-Purpose NL-to-SQL

The Talk-to-Data assistant answers **any reasonable analytical question that can be derived from the approved database schema**. It is not restricted to a hardcoded list of questions.

### Processing Pipeline & Architecture

```
User Analytical Question
           │
           ▼
[Intent & Domain Safety Analysis] ──► (Individual underwriting query ──► Redirect to ML module)
           │
           ▼
[Deterministic Shortcut Check] ────► (If common pattern matches ──► Fast-path SQL template)
           │
           ▼ (General Analytical Questions)
[Schema-Aware LLM Generation] ─────► (Groq / OpenRouter, Temperature = 0, Structured JSON)
           │
           ▼
[Multi-Layer SQL Validator] ───────► (SELECT-only, table whitelist, join checks, EXPLAIN)
           │
           ▼
[Read-Only SQLite Execution] ──────► (data/credit_risk_analytics.db?mode=ro)
           │
           ▼
[Grounded Executive Response] ─────► (Business summary synthesized strictly from result rows)
```

- **Fast Path vs. General Path**: Regex-based deterministic shortcuts serve as an instant, zero-latency fast path for common questions. All other schema-supported analytical queries route dynamically to the schema-aware LLM path.

---

## 12. Relational Database Capability & SQL Security Controls

### Approved Relational Tables
Controlled applicant-level relational queries are supported across five normalized analytical summary tables:
- **`applicants`**: Core demographic, income, credit amount, annuity, and external bureau scores.
- **`bureau_summary`**: Total credit bureau record count and aggregate active bureau debt.
- **`previous_application_summary`**: Historical Home Credit application count and refusal rates.
- **`installment_summary`**: Cumulative installment payment amounts and late-payment delinquency frequency.
- **`applicant_risk_scores`**: CatBoost-generated default probabilities, risk scores, and risk bands.
- **`applicant_analytics`**: Unified 31-column analytics table retained for backward compatibility.

### Multi-Layer Security Controls
- **SELECT-Only Enforcement**: Query validator strictly enforces `SELECT` statements via SQL AST and keyword validation.
- **DDL/DML Blocking**: Blocks `DROP`, `DELETE`, `UPDATE`, `INSERT`, `ALTER`, `ATTACH`, `DETACH`, `CREATE`, `REPLACE`, and `PRAGMA`.
- **Single-Statement Enforcement**: Blocks semicolons and multi-statement execution payloads.
- **SQLite System Table Blocking**: Blocks access to `sqlite_master`, `sqlite_temp_master`, `sqlite_sequence`, and metadata tables.
- **Table Whitelist Enforcement**: Queries can only reference the approved table whitelist.
- **Controlled Join Key Validation**: Multi-table `JOIN` conditions are strictly restricted to primary key `sk_id_curr`.
- **Read-Only SQLite Connection**: SQLite connections are opened using `file:...analytics.db?mode=ro`.
- **EXPLAIN Dry-Run**: Queries are pre-validated using SQLite `EXPLAIN` before execution.
- **Domain Guard & Underwriting Redirection**: Non-analytical queries and individual approval questions (*"Should we approve applicant 100002?"*) are redirected to appropriate modules.
- **Error Masking**: Generic, non-revealing error messages prevent database schema or internal path leakage.

---

## 13. Prompt Engineering and Token Optimization

### Prompt Engineering Architecture
1. **Schema-Aware System Prompt**: Prompts provide exact table schemas, column data types, business descriptions, primary join keys (`sk_id_curr`), aggregate functions, and strict JSON output formatting.
2. **Current Question Isolation**: Active user questions are isolated within `CURRENT QUESTION:` tags to prevent instruction injection.
3. **Structured JSON Output**: The LLM responds in strict JSON (`{"intent": "...", "sql": "...", "reason": "..."}`), eliminating markdown wrapping and conversational fluff.
4. **Grounded Response Generation**: The narrative synthesizer receives only the executed tabular result rows, generating factual executive summaries without hallucinations.

### Token Optimization & Efficiency
- **Compact Schema Representation**: Only schema definitions and column metadata are passed to the model—no raw database rows are transmitted to external LLMs.
- **Deterministic Pattern Bypass**: Standard single-table and relational questions are resolved locally via deterministic regex matching, avoiding external API token usage.
- **Bounded Conversation Context**: Session history is capped using an in-memory LRU cache.
- **Input Character Limits**: Chat input is capped at 2,000 characters to prevent buffer and token exhaustion attacks.
- **Zero-Temperature Decoding**: All SQL generation calls use `temperature: 0.0` for deterministic, reproducible SQL synthesis.

---

## 14. Step-by-Step Setup, Deployment and Operations

### 14.1 Prerequisites
- **Python**: 3.10, 3.11, or 3.12 + `pip` (for local non-Docker backend execution)
- **Node.js**: 18.x or 20.x + `npm` (for local non-Docker frontend development)
- **Docker**: Docker Desktop v24+ with Docker Compose v2+ (for containerized deployment)

> [!NOTE]
> **Docker-Only Execution**: If running the application entirely via Docker, host Python and Node.js installations are **not** required. The Docker containers encapsulate all necessary runtime environments.

---

### 14.2 What Is Included (Processed Runtime Assets)
The repository is distributed with all processed runtime assets necessary to immediately launch and evaluate the application:
- `artifacts/catboost_credit_risk_model.cbm` (2.23 MB): Trained CatBoost binary classification model.
- `artifacts/feature_columns.pkl` (4.83 KB): Ordered list of 231 feature columns for inference alignment.
- `artifacts/model_metadata.json` (469 B): Model performance metrics, operating threshold ($0.66$), and training parameters.
- `artifacts/risk_config.json` (531 B): Decision threshold rules, risk band calibrations, and policy recommendations.
- `data/credit_risk_analytics.db` (261.46 MB): Normalized SQLite analytical data warehouse with 307,511 applicants across 5 relational tables + `applicant_risk_scores`.
- `data/model_features.db` (285.91 MB): High-speed compact SQLite inference feature store containing precomputed feature vectors for all 307,511 applicants.
- `notebooks/credit-risk-intelligence-platform-eda-ml.ipynb` (1.32 MB): Complete exploratory data analysis and ML model training evidence notebook.
- `sql/schema.sql` and `sql/sample_queries.sql`: Analytical database DDL definitions and reference SQL queries.

---

### 14.3 What Is NOT Included (Raw Dataset Policy)
- The original raw Home Credit Default Risk CSV files (~2.64 GB total: `application_train.csv`, `bureau.csv`, `bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`, `credit_card_balance.csv`, `installments_payments.csv`) are **intentionally excluded** from this repository.
- **Why?** Raw CSVs were used offline during feature engineering and database preparation. The live application queries the pre-built, indexed SQLite databases and serialized model artifacts.
- **Action Required**: None. Evaluators should **not** download or place the 2.64 GB raw CSV dataset into the repository to run the application.

---

### 14.4 Runtime Database & Artifact Requirements
Before starting the backend or Docker stack, verify the following files exist in their respective directories:
- `data/credit_risk_analytics.db`
- `data/model_features.db`
- `artifacts/catboost_credit_risk_model.cbm`
- `artifacts/feature_columns.pkl`
- `artifacts/model_metadata.json`
- `artifacts/risk_config.json`

*(These files are already included in the clean repository distribution. There is no need to regenerate them).*

---

### 14.5 Environment Configuration

#### Step 1 — Copy the Configuration Template
From the project root:

- **Windows PowerShell**:
  ```powershell
  Copy-Item .env.example .env
  ```
- **macOS / Linux**:
  ```bash
  cp .env.example .env
  ```

#### Step 2 — Configure Environment Variables in `.env`
Edit `.env` to configure your preferred LLM provider for Talk-to-Data:
```env
# Application Environment
APP_ENV=development
LOG_LEVEL=INFO

# ML Artifact & Database Paths (Default relative paths)
MODEL_PATH=artifacts/catboost_credit_risk_model.cbm
FEATURE_COLUMNS_PATH=artifacts/feature_columns.pkl
MODEL_METADATA_PATH=artifacts/model_metadata.json
RISK_CONFIG_PATH=artifacts/risk_config.json
DATABASE_PATH=data/credit_risk_analytics.db
FEATURE_STORE_PATH=data/model_features.db

# LLM Provider Configuration for Talk-to-Data: 'groq' or 'openrouter'
LLM_PROVIDER=groq
LLM_TEMPERATURE=0.0

# Option 1: Groq Configuration (Recommended for sub-second responses)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Option 2: OpenRouter Configuration (Alternative)
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
```

> [!IMPORTANT]
> - **Security Notice**: Never commit `.env` to version control. `.env` is ignored in `.gitignore` and `.dockerignore`.
> - **Backend-Only Keys**: API keys are consumed strictly by the backend Python service. Never expose them to React/frontend code.
> - **Offline Fallback**: If no API key is configured, Talk-to-Data automatically operates using its built-in offline deterministic fallback engine.

---

### 14.6 Option A: Local Non-Docker Development

#### 1. Backend Setup
From the project root:
```bash
# Create Python virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Start FastAPI backend service
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
- **Backend Service Root**: `http://localhost:8000`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`
- **Backend Health Check**: `http://localhost:8000/health`

*(Note: These URLs apply specifically to non-Docker local backend execution).*

#### 2. Frontend Setup
Because `client/node_modules/` was removed during repository cleanup, restore dependencies before running:
```bash
cd client
npm install
npm run dev
```
- The Vite development server starts at `http://localhost:3000` and automatically proxies `/api/*` and `/health` requests to `http://127.0.0.1:8000`.
- For standalone frontend development against custom endpoints, refer to `client/.env.example` (`VITE_API_BASE_URL`).
- Return to the project root:
```bash
cd ..
```

#### 3. Full-Stack Local Development
Run the two processes simultaneously in separate terminals:
- **Terminal 1 (Backend)**: `python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000`
- **Terminal 2 (Frontend)**: `cd client && npm run dev`
- **Application URL**: Open **[http://localhost:3000](http://localhost:3000)** in your browser.

---

### 14.7 Option B: Docker Containerized Deployment (Recommended)

From the project root:

#### 1. Verify Environment File
Ensure `.env` exists (`cp .env.example .env` or `Copy-Item .env.example .env`).

#### 2. Build and Start Multi-Container Stack
```bash
docker compose up --build -d
```

#### 3. Check Container Health
```bash
docker compose ps
```
Both `credit-risk-backend` and `credit-risk-frontend` should report status `Up (healthy)`.

#### 4. Follow Container Logs
```bash
docker compose logs -f
```

#### 5. Access the Application
- Open **[http://localhost:3000](http://localhost:3000)** in your browser.

> [!NOTE]
> **Docker Port Architecture**:
> - The frontend is exposed on host port `3000`.
> - Backend port `8000` is **internal** to the private Docker network (`credit-risk-net`) for security.
> - Nginx reverse-proxies all `/api/*` and `/health` requests directly to `http://backend:8000/`.
> - Users and browser clients should access `http://localhost:3000` for all application features.

---

### 14.8 Docker Lifecycle Commands

- **Stop All Containers**:
  ```bash
  docker compose down
  ```
- **Restart Without Rebuilding**:
  ```bash
  docker compose up -d
  ```
- **Rebuild Containers After Code or Config Changes**:
  ```bash
  docker compose up --build -d
  ```

---

### 14.9 Verifying the Application After Startup

After opening `http://localhost:3000`, verify core functionality:

1. **Dashboard (`/`)**:
   - Check that portfolio KPIs (307,511 Total Applicants, 8.07% Default Rate, etc.) load immediately.
   - Verify all 6 interactive EDA charts render data cleanly.
2. **Risk Analysis (`/risk-analysis`)**:
   - **Test Low-Risk Applicant `396899`**:
     - Predicted Default Probability $\approx 0.228251$ ($22.83\%$)
     - Risk Score $\approx 22.83 / 100$
     - Risk Band = `Low`
     - Recommendation = `Approve / Low Risk`
   - **Test High-Risk Applicant `100002`**:
     - Predicted Default Probability $\approx 0.827562$ ($82.76\%$)
     - Risk Score $\approx 82.76 / 100$
     - Risk Band = `High`
     - Recommendation = `High Risk / Reject Recommendation`
3. **TreeSHAP Explainability Waterfall**:
   - Verify top positive (risk-increasing) and negative (risk-mitigating) factors are displayed with clear observed values and feature attributions.
4. **Talk-to-Data Assistant (`/chat`)**:
   - Ask: *"What is the default rate?"* $\rightarrow$ Returns 8.07% with SQL explanation.
   - Ask: *"Show me the top 5 high-risk applicants."* $\rightarrow$ Returns top applicants from `applicant_risk_scores`.

---

### 14.10 Build-Time vs. Runtime Data Architecture
- **Offline / Data Preparation Phase**:
  - Used raw Home Credit CSVs to construct normalized analytical tables, pre-engineer 231 features, train the CatBoost model, and populate `applicant_risk_scores`.
- **Runtime / Demonstration Phase**:
  - Requires **only** the pre-built SQLite databases (`data/`), model artifacts (`artifacts/`), backend/frontend code, and configured LLM provider.
  - Raw CSVs are not required at runtime.

---

### 14.11 Optional: Rebuilding Databases From Raw Data
If reproducing the offline data preparation pipeline from scratch:
1. Obtain the Home Credit Default Risk raw CSV files from Kaggle.
2. Set the environment variable to your local raw dataset directory:
   ```bash
   export HOME_CREDIT_DATA_DIR=<path-to-home-credit-data>
   ```
3. Execute the data preparation and feature store build scripts:
   ```bash
   python src/talk_to_data/build_database.py
   python src/ml/build_feature_store.py
   python src/talk_to_data/build_risk_scores.py
   ```
*(Note: This step is entirely optional and unnecessary for running or evaluating the application).*

---

## 15. Security Hardening and Verification Testing

### 15.1 Automated Test Execution
The platform includes comprehensive unit, integration, and security test coverage:

```bash
# Run the complete automated test suite (100 tests)
python -m unittest discover -s src -p "test_*.py"

# Test production frontend build
cd client && npm install && npm run build && cd ..

# Verify Docker container build
docker compose build
docker compose up -d
docker compose ps
```

### 15.2 Validation Test Results
- **Full Backend Test Discovery**: **100 / 100 tests passed** (`Ran 100 tests in 54.085s - OK`).
- **Security Regression Suite**: **34 / 34 tests passed** (SQL injection, prompt injection, sensitive data leakage, underwriting redirection).
- **Talk-to-Data Regression Suite**: **22 / 22 queries passed** across deterministic shortcuts and schema-aware queries.
- **Frontend Production Build**: `npm run build` completed cleanly (582 KB JS, 26 KB CSS).
- **Multi-Container Stack**: Both `credit-risk-backend` and `credit-risk-frontend` verified healthy in Docker.
- **Benchmark Consistency**: Verified 100% numerical agreement between live CatBoost predictor, feature store, `applicant_risk_scores` table, Docker API, and SHAP explainer for benchmark applicants (`396899` and `100002`).

### 15.3 Security Hardening Measures Implemented
- **No Path Leakage**: Generic error responses prevent exposing server directory paths or filesystem layouts.
- **Safe Exception Handling**: Stack traces and raw database errors are masked behind clean user-facing error messages.
- **LRU Session Bounding**: Talk-to-Data chat sessions are managed in an in-memory LRU cache capped at 1,000 active sessions.
- **Strict Payload Limits**: Pydantic models enforce a 2,000-character maximum on incoming chat queries.
- **Explicit CORS Configuration**: Cross-Origin Resource Sharing is configured with explicit HTTP methods and headers.
- **Backend Isolation**: Backend port `8000` is internal to the Docker network, reachable externally only through Nginx.
- **Read-Only Database URIs**: SQLite databases are mounted and queried in read-only mode (`?mode=ro`).
- *Status*: Security-hardened for the intended academic/demo deployment.

---

## 16. Known Limitations, Future Improvements, Operations & Submission

### 16.1 Known Limitations
1. **Model & Validation Scope**:
   - The CatBoost model was evaluated on a single stratified 80/20 train/validation split.
   - The optimal threshold of `0.66` was selected on the validation set to maximize F1 for prototype demonstration; production lending policies require regulatory cutoff calibration.
   - Model predictions represent statistical risk estimates, not guaranteed outcomes. The model is a decision-support prototype and does not act as an autonomous lending authority.
2. **Dataset & Historical Shift**:
   - The Home Credit dataset reflects historical loan applications; borrower behavior and macroeconomic conditions may change over time.
3. **Explainability Bounds**:
   - SHAP values measure mathematical feature attribution within the model and do not constitute real-world causal proof.
   - Excluding protected attributes from user-facing SHAP visualizations promotes compliance but does not replace formal statistical fairness auditing.
4. **Talk-to-Data Boundaries**:
   - The NL-to-SQL engine is strictly restricted to the approved database schema and cannot query external datasets or execute arbitrary Python code.
   - External LLM API calls depend on third-party provider uptime and latency (mitigated by deterministic offline fallback).
5. **Deployment & Access Control**:
   - The prototype does not include user login or Role-Based Access Control (RBAC).

### 16.2 Possible Future Improvements
- **Probability Calibration**: Implementation of Platt Scaling or Isotonic Regression to ensure output probabilities match empirical default frequencies across all deciles.
- **Algorithmic Fairness Audits**: Comprehensive fairness evaluations using metrics such as Disparate Impact Ratio, Equalized Odds, and Demographic Parity.
- **Model Drift & Monitoring**: Integration of automated drift detection tools (e.g., Evidently AI) to monitor feature drift and concept drift in production.
- **Authentication & RBAC**: Integration of OAuth2 / JWT authentication with role-based access for Underwriters, Risk Managers, and Compliance Auditors.
- **Secrets Management**: Integration with enterprise secret vaults (e.g., AWS Secrets Manager, HashiCorp Vault) for production credential rotation.
- **Real-Time Streaming**: Server-Sent Events (SSE) streaming for LLM narrative responses.

### 16.3 Troubleshooting Guide

| Issue | Likely Cause | Resolution |
|---|---|---|
| **Docker containers fail to start** | Port 3000 already occupied or daemon not running | Run `docker compose ps` and `docker compose logs -f`. Ensure host port 3000 is free. |
| **Missing database error on startup** | Database files missing from `data/` | Verify `data/credit_risk_analytics.db` (261.46 MB) and `data/model_features.db` (285.91 MB) exist. |
| **Missing model artifact error** | Files missing from `artifacts/` | Verify all four files in `artifacts/` (`catboost_credit_risk_model.cbm`, `feature_columns.pkl`, `model_metadata.json`, `risk_config.json`) exist. |
| **Talk-to-Data uses deterministic fallback** | API key not provided or invalid | Check `LLM_PROVIDER`, `GROQ_API_KEY`, or `OPENROUTER_API_KEY` in `.env`. Ensure `.env` is in the project root. |
| **Frontend cannot connect to backend (Docker)** | Browser attempting to reach port 8000 | Always access `http://localhost:3000`. Nginx automatically proxies API requests to internal backend port 8000. |
| **Frontend cannot connect (Local Non-Docker)** | FastAPI backend not running | Ensure Uvicorn is running on port 8000 in a separate terminal. |
| **Frontend npm build or dependency errors** | Missing `node_modules/` | Run `cd client && npm install` to restore dependencies from `package-lock.json`. Do not delete lockfiles. |

### 16.4 Preparing the Project for Submission / Archive

#### Files to Include in Submission Archive:
- `src/` (All backend source code, routes, ML engines, rules, and test files)
- `client/src/` (React frontend source code, components, pages, CSS)
- `client/dist/` (Pre-compiled production frontend bundle required by `client/Dockerfile`)
- `client/Dockerfile`, `client/nginx.conf`, `client/package.json`, `client/package-lock.json`, `client/vite.config.js`, `client/tailwind.config.js`, `client/postcss.config.js`, `client/index.html`
- `artifacts/` (Trained model, feature index, model metadata, risk config)
- `data/` (`credit_risk_analytics.db`, `model_features.db`, `.gitkeep`)
- `notebooks/` (`credit-risk-intelligence-platform-eda-ml.ipynb`)
- `sql/` (`schema.sql`, `sample_queries.sql`)
- `Dockerfile`, `docker-compose.yml`, `requirements.txt`, `README.md`, `.env.example`, `.gitignore`, `.dockerignore`
- Assignment PDF specifications (`NeoStats_*.pdf`, optional reference documents)

#### Files to Explicitly Exclude from Submission:
- `.env` (Never commit or distribute real API keys or private credentials)
- Raw Home Credit CSV files (2.64 GB raw dataset is excluded per assignment rules)
- `client/node_modules/` (Restored automatically via `npm install`)
- `__pycache__/` and `*.pyc` files
- Temporary logs, debug scripts, or IDE scratch directories

---

### 16.5 Project Directory Tree

```
credit-risk-intelligence/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── README.md
├── NeoStats_AI_Use_Case.pdf
├── NeoStats_Candidate_Assignment.pdf
│
├── artifacts/
│   ├── catboost_credit_risk_model.cbm        [2.23 MB Model Binary]
│   ├── feature_columns.pkl                   [4.83 KB Feature Alignment Index]
│   ├── model_metadata.json                   [469 B Model Hyperparameters & Threshold]
│   └── risk_config.json                      [531 B Risk Bands & Decision Policies]
│
├── client/
│   ├── .dockerignore
│   ├── .env.example
│   ├── Dockerfile                           [Nginx Production Container]
│   ├── index.html
│   ├── nginx.conf                           [Reverse Proxy Configuration]
│   ├── package.json
│   ├── package-lock.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── vite.config.js
│   ├── dist/                                [Pre-compiled Frontend Assets]
│   │   ├── index.html
│   │   └── assets/
│   │       ├── index-*.js
│   │       └── index-*.css
│   └── src/
│       ├── App.jsx
│       ├── index.css
│       ├── main.jsx
│       ├── components/
│       │   ├── ChatInput.jsx
│       │   ├── ChatMessage.jsx
│       │   ├── ErrorMessage.jsx
│       │   ├── InsightChart.jsx
│       │   ├── LoadingState.jsx
│       │   ├── Navbar.jsx
│       │   ├── RiskGauge.jsx
│       │   ├── RiskResult.jsx
│       │   ├── ShapFactors.jsx
│       │   └── SummaryCard.jsx
│       ├── pages/
│       │   ├── Dashboard.jsx
│       │   ├── RiskAnalysis.jsx
│       │   └── TalkToData.jsx
│       └── services/
│           └── api.js
│
├── data/
│   ├── .gitkeep
│   ├── credit_risk_analytics.db              [261.46 MB SQLite Data Warehouse]
│   └── model_features.db                     [285.91 MB Inference Feature Store]
│
├── documents/
│   └── .gitkeep
│
├── notebooks/
│   └── credit-risk-intelligence-platform-eda-ml.ipynb [1.32 MB EDA & Training Notebook]
│
├── sql/
│   ├── sample_queries.sql
│   └── schema.sql
│
└── src/
    ├── api/
    │   ├── README.md
    │   ├── __init__.py
    │   ├── dependencies.py
    │   ├── main.py
    │   ├── schemas.py
    │   ├── test_api.py                      [FastAPI Route Test Suite]
    │   └── routes/
    │       ├── __init__.py
    │       ├── chat.py
    │       ├── eda.py
    │       └── ml.py
    ├── ml/
    │   ├── __init__.py
    │   ├── build_feature_store.py
    │   ├── explainer.py
    │   ├── feature_store.py
    │   ├── predictor.py
    │   └── test_inference_store.py          [Feature Store & Model Test Suite]
    ├── rules/
    │   ├── __init__.py
    │   └── risk_rules.py
    ├── talk_to_data/
    │   ├── __init__.py
    │   ├── build_database.py
    │   ├── build_risk_scores.py
    │   ├── chatbot.py
    │   ├── database.py
    │   ├── llm.py
    │   ├── prompts.py
    │   ├── query_engine.py
    │   ├── response_generator.py
    │   ├── schema.py
    │   ├── sql_generator.py
    │   ├── sql_validator.py
    │   ├── test_general_schema_queries.py   [General NL-to-SQL Test Suite]
    │   ├── test_llm_providers.py            [LLM Failover Test Suite]
    │   ├── test_relational_queries.py       [Relational Query Test Suite]
    │   ├── test_talk_to_data.py             [Intent & Security Test Suite]
    │   └── test_talk_to_data_e2e.py         [E2E Conversation Test Suite]
    └── utils/
        └── .gitkeep
```

---

> **Final Note:** This project is an AI-assisted credit risk decision-support prototype. Its ML predictions, explanations, business rules, and conversational analytics are intended to support human analysis and evaluation rather than replace human underwriting decision-making.

