"""
ML Inference Feature Store & Prediction Access Layer
====================================================
Provides low-latency, read-only retrieval of the exact 231 pre-engineered model features
from `data/model_features.db` for runtime CatBoost prediction and SHAP explanation.

Guarantees 100% offline inference without requiring the raw 2.64 GB Home Credit dataset.
"""

import logging
import os
import pickle
import sqlite3
from typing import Any, Dict, List, Optional

import pandas as pd

from src.ml.explainer import CreditRiskExplainer
from src.ml.predictor import CreditRiskPredictor

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_STORE_PATH = os.path.join("data", "model_features.db")
DEFAULT_FEATURE_COLUMNS_PATH = os.path.join("artifacts", "feature_columns.pkl")


def load_feature_columns(path: Optional[str] = None) -> List[str]:
    """Load authoritative 231 feature columns list."""
    fpath = path or DEFAULT_FEATURE_COLUMNS_PATH
    with open(fpath, "rb") as f:
        return pickle.load(f)


def get_applicant_count(db_path: Optional[str] = None) -> int:
    """Return total number of applicants stored in the feature store."""
    target_db = db_path or DEFAULT_FEATURE_STORE_PATH
    if not os.path.exists(target_db):
        return 0

    conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM applicant_model_features;")
        count = cur.fetchone()[0]
        return int(count)
    finally:
        conn.close()


def get_applicant_features(
    applicant_id: int,
    db_path: Optional[str] = None,
    feature_columns: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    Retrieve the exact 231 model features for a given applicant ID in read-only mode.

    Args:
        applicant_id: SK_ID_CURR identifier.
        db_path: Path to model_features.db.
        feature_columns: Optional list of authoritative feature columns.

    Returns:
        Single-row pandas DataFrame containing the exact 231 features, or None if not found.
    """
    target_db = db_path or DEFAULT_FEATURE_STORE_PATH
    if not os.path.exists(target_db):
        logger.error("Feature store not found at: %s", target_db)
        return None

    cols = feature_columns or load_feature_columns()

    conn = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
    try:
        query = "SELECT * FROM applicant_model_features WHERE SK_ID_CURR = ? LIMIT 1;"
        df = pd.read_sql_query(query, conn, params=(int(applicant_id),))
        if df.empty:
            return None

        # Reindex to exact 231 authoritative features in exact order
        features_df = df.reindex(columns=cols)
        return features_df
    finally:
        conn.close()


def predict_applicant(
    applicant_id: int,
    predictor: Optional[CreditRiskPredictor] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Predict default risk and generate decision support for an applicant ID using the feature store.

    Args:
        applicant_id: SK_ID_CURR integer.
        predictor: Optional CreditRiskPredictor instance.
        db_path: Optional custom path to model_features.db.

    Returns:
        Structured prediction dictionary with status, probability, score, band, and decision.
    """
    try:
        app_id_int = int(applicant_id)
    except (ValueError, TypeError):
        return {
            "success": False,
            "applicant_id": applicant_id,
            "error": "Invalid applicant ID format. Must be an integer."
        }

    features = get_applicant_features(app_id_int, db_path=db_path)
    if features is None or features.empty:
        return {
            "success": False,
            "applicant_id": app_id_int,
            "error": f"Applicant ID {app_id_int} not found in the inference feature store."
        }

    pred_engine = predictor or CreditRiskPredictor()
    pred_result = pred_engine.predict(features)

    return {
        "success": True,
        "applicant_id": app_id_int,
        "default_probability": pred_result["Default_Probability"],
        "risk_score": pred_result["Risk_Score"],
        "risk_band": pred_result["Risk_Band"],
        "prediction": pred_result["Predicted_Default"],
        "decision": pred_result["Decision"]
    }


def explain_applicant(
    applicant_id: int,
    explainer: Optional[CreditRiskExplainer] = None,
    predictor: Optional[CreditRiskPredictor] = None,
    db_path: Optional[str] = None,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Generate SHAP-based feature importance explanation for an applicant ID from the feature store.

    Args:
        applicant_id: SK_ID_CURR integer.
        explainer: Optional CreditRiskExplainer instance.
        predictor: Optional CreditRiskPredictor instance.
        db_path: Optional custom path to model_features.db.
        top_k: Number of top contributing features to return.

    Returns:
        Structured explanation dictionary with base value, top risk factors, and mitigated factors.
    """
    try:
        app_id_int = int(applicant_id)
    except (ValueError, TypeError):
        return {
            "success": False,
            "applicant_id": applicant_id,
            "error": "Invalid applicant ID format. Must be an integer."
        }

    features = get_applicant_features(app_id_int, db_path=db_path)
    if features is None or features.empty:
        return {
            "success": False,
            "applicant_id": app_id_int,
            "error": f"Applicant ID {app_id_int} not found in the inference feature store."
        }

    pred_engine = predictor or CreditRiskPredictor()
    shap_engine = explainer or CreditRiskExplainer(predictor=pred_engine)

    explanation = shap_engine.explain_applicant(features, top_n=top_k)
    explanation["success"] = True
    explanation["applicant_id"] = app_id_int
    return explanation
