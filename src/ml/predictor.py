"""
Credit Risk Predictor Module
============================
Provides reusable machine learning inference for loan default probability,
0-100 risk scoring, Low/Medium/High risk banding, and business decision support.

Loads the trained CatBoost model and feature schema from artifacts/ without
relying on notebook state.
"""

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from src.rules.risk_rules import assign_decision, assign_risk_band, load_risk_config

DEFAULT_MODEL_PATH = os.path.join("artifacts", "catboost_credit_risk_model.cbm")
DEFAULT_FEATURES_PATH = os.path.join("artifacts", "feature_columns.pkl")
DEFAULT_METADATA_PATH = os.path.join("artifacts", "model_metadata.json")
DEFAULT_RISK_CONFIG_PATH = os.path.join("artifacts", "risk_config.json")


class CreditRiskPredictor:
    """
    Production-ready inference engine for the CatBoost credit default prediction model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_columns_path: Optional[str] = None,
        risk_config_path: Optional[str] = None,
        metadata_path: Optional[str] = None
    ) -> None:
        """
        Initialize the predictor by loading all necessary model artifacts and configurations.
        """
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.feature_columns_path = feature_columns_path or DEFAULT_FEATURES_PATH
        self.risk_config_path = risk_config_path or DEFAULT_RISK_CONFIG_PATH
        self.metadata_path = metadata_path or DEFAULT_METADATA_PATH

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load trained CatBoost model, feature list, and risk configuration."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not os.path.exists(self.feature_columns_path):
            raise FileNotFoundError(f"Feature columns file not found at: {self.feature_columns_path}")

        # Load CatBoost Model
        self.model = CatBoostClassifier()
        self.model.load_model(self.model_path)

        # Load Feature Columns List
        with open(self.feature_columns_path, "rb") as f:
            self.feature_columns: List[str] = pickle.load(f)

        # Load Risk Configuration
        self.risk_config: Dict[str, Any] = load_risk_config(self.risk_config_path)
        self.classification_threshold: float = float(
            self.risk_config.get("classification_threshold", 0.66)
        )

        # Load Model Metadata (optional / informative)
        self.metadata: Dict[str, Any] = {}
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Identify categorical feature column names from the CatBoost model
        cat_indices = self.model.get_cat_feature_indices()
        self.categorical_features: List[str] = [
            self.feature_columns[idx] for idx in cat_indices if idx < len(self.feature_columns)
        ]

    def preprocess_applicant(
        self, applicant_data: Union[pd.DataFrame, pd.Series, Dict[str, Any]]
    ) -> (Optional[Union[str, int]], pd.DataFrame):
        """
        Apply identical feature engineering, ratio calculations, and column alignments
        matching the notebook's final pipeline.

        Args:
            applicant_data: Applicant feature dictionary, pandas Series, or single-row DataFrame.

        Returns:
            Tuple of (applicant_id, preprocessed_dataframe_with_231_features)
        """
        if isinstance(applicant_data, dict):
            df = pd.DataFrame([applicant_data])
        elif isinstance(applicant_data, pd.Series):
            df = applicant_data.to_frame().T
        elif isinstance(applicant_data, pd.DataFrame):
            df = applicant_data.copy()
        else:
            raise TypeError("applicant_data must be a dict, pd.Series, or pd.DataFrame")

        # Extract Applicant ID if present
        applicant_id = None
        for id_col in ["SK_ID_CURR", "Applicant_ID", "applicant_id", "id", "ID"]:
            if id_col in df.columns:
                applicant_id = df[id_col].iloc[0]
                df = df.drop(columns=[id_col])
                break

        # If applicant already contains pre-engineered columns and matches feature list, proceed
        # Otherwise compute engineered features if source raw columns are present
        if "AMT_INCOME_TOTAL" in df.columns:
            income = df["AMT_INCOME_TOTAL"].replace(0, np.nan)
            if "AMT_CREDIT" in df.columns:
                df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / income
                df["ANNUITY_CREDIT_RATIO"] = df.get("AMT_ANNUITY", np.nan) / df["AMT_CREDIT"].replace(0, np.nan)
            if "AMT_ANNUITY" in df.columns:
                df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / income
            if "AMT_GOODS_PRICE" in df.columns:
                df["GOODS_INCOME_RATIO"] = df["AMT_GOODS_PRICE"] / income
                if "AMT_CREDIT" in df.columns:
                    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)

        if "DAYS_BIRTH" in df.columns and "AGE_YEARS" not in df.columns:
            df["AGE_YEARS"] = (-df["DAYS_BIRTH"]) / 365.25

        if "DAYS_EMPLOYED" in df.columns:
            df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
            if "EMPLOYMENT_YEARS" not in df.columns:
                df["EMPLOYMENT_YEARS"] = (-df["DAYS_EMPLOYED"]) / 365.25

        if "AMT_INCOME_TOTAL" in df.columns:
            if "CNT_FAM_MEMBERS" in df.columns:
                df["INCOME_PER_FAMILY_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].replace(0, np.nan)
            if "CNT_CHILDREN" in df.columns:
                df["INCOME_PER_CHILD"] = df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)

        # Document count
        doc_cols = [c for c in df.columns if c.startswith("FLAG_DOCUMENT_")]
        if doc_cols and "DOCUMENTS_PROVIDED_COUNT" not in df.columns:
            df["DOCUMENTS_PROVIDED_COUNT"] = df[doc_cols].sum(axis=1)

        # Historical flags
        for col_name, flag_name in [
            ("CC_CONTRACT_COUNT", "HAS_CC_HISTORY"),
            ("BUREAU_CREDIT_COUNT", "HAS_BUREAU_HISTORY"),
            ("PREV_APP_COUNT", "HAS_PREVIOUS_APPLICATION"),
            ("POS_CONTRACT_COUNT", "HAS_POS_HISTORY"),
            ("INSTALLMENT_COUNT", "HAS_INSTALLMENT_HISTORY")
        ]:
            if col_name in df.columns and flag_name not in df.columns:
                df[flag_name] = df[col_name].notna().astype("int8")

        # Bureau ratios
        if "BUREAU_TOTAL_CREDIT" in df.columns:
            credit = df["BUREAU_TOTAL_CREDIT"].replace(0, np.nan)
            if "BUREAU_TOTAL_DEBT" in df.columns:
                df["BUREAU_DEBT_CREDIT_RATIO"] = df["BUREAU_TOTAL_DEBT"] / credit
            if "BUREAU_TOTAL_OVERDUE" in df.columns:
                df["BUREAU_OVERDUE_CREDIT_RATIO"] = df["BUREAU_TOTAL_OVERDUE"] / credit

        # Replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)

        # Reindex to exact 231 feature columns
        df = df.reindex(columns=self.feature_columns)

        # Format categorical columns
        for col in self.categorical_features:
            if col in df.columns:
                df[col] = df[col].fillna("Missing").astype(str)

        return applicant_id, df

    def predict(
        self, applicant_data: Union[pd.DataFrame, pd.Series, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate complete credit risk prediction, scoring, risk band, and decision recommendation.

        Args:
            applicant_data: Dict, Series, or single-row DataFrame.

        Returns:
            Dictionary with Applicant_ID, Default_Probability, Risk_Score, Risk_Band,
            Predicted_Default, and Decision.
        """
        applicant_id, processed_df = self.preprocess_applicant(applicant_data)

        # Predict probability of default (Class 1)
        proba_array = self.model.predict_proba(processed_df)
        default_prob = float(proba_array[0][1])

        # Risk score (0 - 100)
        risk_score = round(default_prob * 100, 2)

        # Risk band from config
        risk_band = assign_risk_band(default_prob, self.risk_config)

        # Classification decision based on tuned threshold
        predicted_default = int(default_prob >= self.classification_threshold)

        # Standardized business decision recommendation
        decision = assign_decision(risk_band, self.risk_config)

        return {
            "Applicant_ID": applicant_id,
            "Default_Probability": default_prob,
            "Risk_Score": risk_score,
            "Risk_Band": risk_band,
            "Predicted_Default": predicted_default,
            "Decision": decision
        }
