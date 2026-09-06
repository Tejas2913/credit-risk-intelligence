"""
Pydantic Data Models & Request/Response Schemas
================================================
Defines strict schemas for EDA, ML inference, SHAP explanation, Talk-to-Data, and Health endpoints.
"""

from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


# ----------------------------------------------------------------------
# Health & Status Schemas
# ----------------------------------------------------------------------

class DatabaseHealth(BaseModel):
    analytics_db: bool = Field(..., description="Whether credit_risk_analytics.db exists and is queryable")
    feature_store_db: bool = Field(..., description="Whether model_features.db exists and is queryable")


class HealthResponse(BaseModel):
    status: str = Field("ok", description="Overall health status")
    service: str = Field("credit-risk-intelligence-api", description="Service name")
    version: str = Field("1.0.0", description="API version")
    databases: DatabaseHealth = Field(..., description="Status of SQLite analytical and feature stores")
    model_loaded: bool = Field(..., description="Whether the CatBoost model is loaded and ready")
    message: str = Field("Credit Risk Intelligence API is operating normally.", description="Status message")


# ----------------------------------------------------------------------
# Module 1: EDA / Analytics Schemas
# ----------------------------------------------------------------------

class EDASummaryResponse(BaseModel):
    total_applicants: int = Field(..., description="Total number of evaluated applicants in dataset")
    default_applicants: int = Field(..., description="Total number of defaulted applicants (TARGET=1)")
    non_default_applicants: int = Field(..., description="Total number of non-defaulted applicants (TARGET=0)")
    default_rate: float = Field(..., description="Overall baseline default rate (0.0 to 1.0)")
    default_rate_pct: float = Field(..., description="Overall baseline default rate percentage (0.0 to 100.0%)")
    total_features: int = Field(231, description="Total engineered features used by the ML model")
    numerical_features: int = Field(106, description="Number of numerical application features")
    categorical_features: int = Field(16, description="Number of categorical application features")


class InsightChartData(BaseModel):
    labels: List[str] = Field(..., description="Category labels for chart X-axis")
    values: List[float] = Field(..., description="Metric values (percentages or counts) for chart Y-axis")
    counts: Optional[List[int]] = Field(None, description="Sample sizes / observation counts per category")


class EDAInsight(BaseModel):
    id: int = Field(..., description="Insight sequence identifier (1 to 6)")
    title: str = Field(..., description="Business insight title")
    metric: str = Field(..., description="Key headline metric or observation")
    chart: InsightChartData = Field(..., description="Chart-ready data structure for frontend visualization")
    interpretation: str = Field(..., description="Authoritative business context and modeling implications")


class EDAInsightsResponse(BaseModel):
    insights: List[EDAInsight] = Field(..., description="List of the 6 core business insights from EDA")


# ----------------------------------------------------------------------
# Module 2: ML Inference & SHAP/XAI Schemas
# ----------------------------------------------------------------------

class ApplicantRiskResponse(BaseModel):
    success: bool = Field(True, description="Whether prediction succeeded")
    applicant_id: int = Field(..., description="Unique applicant SK_ID_CURR identifier")
    default_probability: float = Field(..., description="Calibrated default probability (0.0 to 1.0)")
    risk_score: float = Field(..., description="Standardized risk score (0 to 100)")
    risk_band: str = Field(..., description="Risk tier: Low, Medium, or High")
    prediction: int = Field(..., description="Binary classification (0 = Non-Default, 1 = Default)")
    decision: str = Field(..., description="Decision-support recommendation: Approve, Review, or Decline")


class SHAPFactor(BaseModel):
    feature: str = Field(..., description="Technical feature column name")
    feature_value: Optional[Union[float, int, str]] = Field(None, description="Applicant's actual feature value")
    shap_value: float = Field(..., description="SHAP contribution value to log-odds")
    direction: str = Field(..., description="'Increases Risk' or 'Reduces Risk'")
    description: Optional[str] = Field(None, description="Human-readable explanation of the feature")


class ApplicantExplanationResponse(BaseModel):
    success: bool = Field(True, description="Whether SHAP explanation succeeded")
    applicant_id: int = Field(..., description="Unique applicant SK_ID_CURR identifier")
    base_value: float = Field(..., description="Model expected value (average base log-odds across training data)")
    risk_increasing_factors: List[SHAPFactor] = Field(..., description="Top features contributing to higher default risk")
    risk_reducing_factors: List[SHAPFactor] = Field(..., description="Top features contributing to lower default risk")
    disclaimer: str = Field(
        "Decision-support explanation only. Protected demographic attributes are excluded from explanations.",
        description="Regulatory & compliance disclaimer"
    )


# ----------------------------------------------------------------------
# Module 3: Talk-to-Data Chat Schemas
# ----------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Natural language question about credit risk portfolio")
    session_id: Optional[str] = Field(None, description="Optional session identifier for conversation continuity")


class ChatResponse(BaseModel):
    success: bool = Field(..., description="Whether query was executed or handled cleanly")
    answer: str = Field(..., description="Business-readable executive answer")
    sql: Optional[str] = Field(None, description="Generated read-only SQL query executed against analytics DB")
    data: List[Dict[str, Any]] = Field(default_factory=list, description="Raw query result rows as list of dicts")
    columns: List[str] = Field(default_factory=list, description="Result column names")
    row_count: int = Field(0, description="Number of rows returned")
    intent: str = Field(..., description="Classified intent (e.g. portfolio_analytics, underwriting_redirect, etc.)")
    used_llm: bool = Field(False, description="Whether an external LLM was used for generation/synthesis")
    error: Optional[str] = Field(None, description="Error message if processing failed")


# ----------------------------------------------------------------------
# Error Schema
# ----------------------------------------------------------------------

class APIErrorResponse(BaseModel):
    success: bool = Field(False, description="Failure status flag")
    error: str = Field(..., description="Concise error summary")
    detail: Optional[str] = Field(None, description="Detailed error description or guidance")
