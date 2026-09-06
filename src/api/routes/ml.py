"""
Module 2: ML Inference & SHAP/XAI Route Handlers
=================================================
Exposes low-latency credit risk prediction and SHAP feature attribution
for individual applicants using `data/model_features.db` and the trained CatBoost model.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.api.dependencies import get_explainer, get_predictor
from src.api.schemas import (
    ApplicantExplanationResponse,
    ApplicantRiskResponse,
    SHAPFactor,
)
from src.ml.explainer import CreditRiskExplainer
from src.ml.feature_store import explain_applicant, predict_applicant
from src.ml.predictor import CreditRiskPredictor

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ML Inference & Explainability"])


@router.get(
    "/applicant/{applicant_id}",
    response_model=ApplicantRiskResponse,
    summary="Predict default risk and decision-support recommendation for an applicant",
    responses={
        404: {"description": "Applicant ID not found in feature store"},
        422: {"description": "Invalid applicant ID format"}
    }
)
def get_applicant_risk(
    applicant_id: int,
    predictor: CreditRiskPredictor = Depends(get_predictor)
) -> ApplicantRiskResponse:
    """
    Retrieve 231 pre-engineered features from the feature store and evaluate default risk.

    Returns:
    - default_probability: Model predicted probability of default (0.0 - 1.0)
    - risk_score: Scaled score (0 - 100)
    - risk_band: Low (<0.30), Medium (0.30-0.60), or High (>=0.60)
    - prediction: Binary classification (0: Non-Default, 1: Default)
    - decision: Decision-support recommendation (Approve, Review, or Decline)
    """
    result = predict_applicant(applicant_id, predictor=predictor)

    if not result.get("success", False):
        error_msg = result.get("error", "Applicant ID not found in the inference feature store.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_msg
        )

    return ApplicantRiskResponse(
        success=True,
        applicant_id=result["applicant_id"],
        default_probability=round(float(result["default_probability"]), 6),
        risk_score=round(float(result["risk_score"]), 2),
        risk_band=str(result["risk_band"]),
        prediction=int(result["prediction"]),
        decision=str(result["decision"])
    )


@router.get(
    "/applicant/{applicant_id}/explanation",
    response_model=ApplicantExplanationResponse,
    summary="Generate SHAP-based feature importance explanation for an applicant",
    responses={
        404: {"description": "Applicant ID not found in feature store"},
        422: {"description": "Invalid applicant ID format"}
    }
)
def get_applicant_explanation(
    applicant_id: int,
    top_k: int = Query(5, ge=1, le=20, description="Number of top positive/negative factors to return"),
    predictor: CreditRiskPredictor = Depends(get_predictor),
    explainer: CreditRiskExplainer = Depends(get_explainer)
) -> ApplicantExplanationResponse:
    """
    Generate explainable AI attribution (SHAP values) for an applicant's prediction.
    Excludes sensitive demographic attributes (gender, marital status, education, car/realty flags)
    from user-facing explanations for fairness and compliance.
    """
    result = explain_applicant(
        applicant_id=applicant_id,
        explainer=explainer,
        predictor=predictor,
        top_k=top_k
    )

    if not result.get("success", False):
        error_msg = result.get("error", "Applicant ID not found in the inference feature store.")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_msg
        )

    increasing = [
        SHAPFactor(
            feature=str(f.get("feature") or f.get("Feature") or ""),
            feature_value=f.get("raw_value") if f.get("raw_value") is not None else f.get("Feature_Value", f.get("Value")),
            shap_value=round(float(f.get("shap_value", f.get("SHAP_Value", 0.0))), 6),
            direction="Increases Risk",
            description=f.get("description") or f.get("Description") or f.get("readable_text")
        )
        for f in result.get("Risk_Increasing_Factors", [])
    ]

    reducing = [
        SHAPFactor(
            feature=str(f.get("feature") or f.get("Feature") or ""),
            feature_value=f.get("raw_value") if f.get("raw_value") is not None else f.get("Feature_Value", f.get("Value")),
            shap_value=round(float(f.get("shap_value", f.get("SHAP_Value", 0.0))), 6),
            direction="Reduces Risk",
            description=f.get("description") or f.get("Description") or f.get("readable_text")
        )
        for f in result.get("Risk_Reducing_Factors", result.get("Risk_Mitigating_Factors", []))
    ]

    return ApplicantExplanationResponse(
        success=True,
        applicant_id=result["applicant_id"],
        base_value=round(float(result.get("Base_Value", 0.0)), 6),
        risk_increasing_factors=increasing,
        risk_reducing_factors=reducing,
        disclaimer=(
            "Decision-support explanation only. Excludes protected demographic attributes "
            "(gender, family status, education, assets) in compliance with fair-lending standards."
        )
    )
