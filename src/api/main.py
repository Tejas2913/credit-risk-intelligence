"""
Unified FastAPI Application Entrypoint
======================================
Coordinates the 3 core platform capabilities:
1. EDA / Business Insights (/api/eda)
2. ML Credit Risk Prediction & SHAP Explainability (/api/ml)
3. Talk-to-Data NL-to-SQL Analytics (/api/chat)

Provides health check, CORS middleware, and OpenAPI documentation.
"""

import logging
import os
from typing import Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import check_db_health, get_predictor
from src.api.routes import chat, eda, ml
from src.api.schemas import DatabaseHealth, HealthResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI application instance
app = FastAPI(
    title="Credit Risk Intelligence Platform API",
    description=(
        "Enterprise-grade decision-support backend for Credit Risk Assessment. "
        "Provides portfolio analytics (EDA), real-time applicant scoring & explainability (SHAP), "
        "and a natural language SQL query assistant (Talk-to-Data). "
        "Designed for decision-support; does not act as an autonomous lending authority."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration for local development
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",
    "http://localhost:8000",
    "http://localhost:8501",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
)

# Register route modules
app.include_router(eda.router)
app.include_router(ml.router)
app.include_router(chat.router)


@app.get("/health", response_model=HealthResponse, tags=["Health & Status"], summary="System health check")
def health_check() -> HealthResponse:
    """
    Verify the operational readiness of the API, databases, and ML model.
    Checks existence of analytical SQLite DB, feature store SQLite DB, and CatBoost model.
    """
    analytics_ok = check_db_health(os.path.join("data", "credit_risk_analytics.db"), "applicant_analytics")
    features_ok = check_db_health(os.path.join("data", "model_features.db"), "applicant_model_features")

    model_ok = False
    try:
        predictor = get_predictor()
        model_ok = predictor.model is not None
    except Exception as exc:
        logger.warning("Predictor health check failed: %s", exc)

    all_healthy = analytics_ok and features_ok and model_ok

    return HealthResponse(
        status="ok" if all_healthy else "degraded",
        service="credit-risk-intelligence-api",
        version="1.0.0",
        databases=DatabaseHealth(
            analytics_db=analytics_ok,
            feature_store_db=features_ok
        ),
        model_loaded=model_ok,
        message="All systems operational and ready for inference." if all_healthy else "Some subsystems are unavailable."
    )


@app.get("/", tags=["Health & Status"], summary="Root endpoint")
def root_endpoint() -> Dict[str, str]:
    """Root redirect / welcome information pointing to interactive API documentation."""
    return {
        "message": "Welcome to Credit Risk Intelligence Platform API",
        "documentation": "/docs",
        "health": "/health"
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global fallback exception handler ensuring clean JSON error responses without stack leaks."""
    logger.error("Unhandled server exception on %s: %s", request.url.path, exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Internal server error occurred.",
            "detail": "An unexpected error occurred while processing your request."
        }
    )
