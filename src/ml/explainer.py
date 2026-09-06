"""
Credit Risk Explainability Module (XAI)
======================================
Provides SHAP-based local model explanations for credit risk predictions.

Converts technical SHAP values into business-readable, non-causal explanations
while filtering out sensitive demographic and protected attributes for end users.
"""

import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shap

from src.ml.predictor import CreditRiskPredictor

# Protected / sensitive features excluded from user-facing explanations
EXCLUDED_EXPLANATION_FEATURES = {
    "CODE_GENDER",
    "NAME_FAMILY_STATUS",
    "NAME_EDUCATION_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY"
}

# Mapping of technical feature names to clear business-readable terminology
FEATURE_DESCRIPTIONS = {
    "EXT_SOURCE_1": "External credit score 1",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_3": "External credit score 3",
    "POS_AVG_FUTURE_INSTALLMENTS": "Average number of future installments",
    "LATE_PAYMENT_RATIO": "Historical late-payment frequency",
    "ANNUITY_CREDIT_RATIO": "Loan annuity relative to credit amount",
    "CREDIT_GOODS_RATIO": "Credit amount relative to goods price",
    "AMT_ANNUITY": "Loan annuity amount",
    "PREV_AVG_CREDIT_APPLICATION_RATIO": "Historical credit-to-application ratio",
    "DAYS_EMPLOYED": "Employment duration",
    "EMPLOYMENT_YEARS": "Employment duration",
    "BUREAU_DEBT_CREDIT_RATIO": "Historical debt relative to credit",
    "TOTAL_PAYMENT_AMOUNT": "Historical total payment amount",
    "PREV_REFUSAL_RATE": "Previous application refusal rate",
    "BUREAU_ACTIVE_COUNT": "Number of active historical credits",
    "BUREAU_AVG_LIMIT": "Average historical credit limit",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "AMT_CREDIT": "Loan credit amount",
    "AMT_INCOME_TOTAL": "Total applicant income",
    "AGE_YEARS": "Applicant age",
    "AMT_REQ_CREDIT_BUREAU_QRT": "Recent credit-bureau inquiries (Quarter)",
    "AMT_REQ_CREDIT_BUREAU_YEAR": "Credit-bureau inquiries (Year)",
    "DEF_30_CNT_SOCIAL_CIRCLE": "Defaults in social circle (30 DPD)",
    "DEF_60_CNT_SOCIAL_CIRCLE": "Defaults in social circle (60 DPD)",
    "DAYS_ID_PUBLISH": "Days since ID document published",
    "DAYS_REGISTRATION": "Days since registration",
    "REGION_RATING_CLIENT_W_CITY": "Region client rating with city",
    "REGION_RATING_CLIENT": "Region client rating",
    "NAME_CONTRACT_TYPE": "Contract type",
    "NAME_INCOME_TYPE": "Income source type",
    "OCCUPATION_TYPE": "Occupation type",
    "ORGANIZATION_TYPE": "Organization type",
    "DOCUMENTS_PROVIDED_COUNT": "Number of documents provided",
    "CREDIT_INCOME_RATIO": "Credit amount relative to total income",
    "ANNUITY_INCOME_RATIO": "Annuity relative to total income",
    "GOODS_INCOME_RATIO": "Goods price relative to total income",
    "INCOME_PER_FAMILY_MEMBER": "Income per family member",
    "INCOME_PER_CHILD": "Income per child",
    "PREV_APP_COUNT": "Count of previous loan applications",
    "INSTALLMENT_COUNT": "Count of historical installments paid"
}


def format_feature_value(feature: str, value: Any) -> str:
    """
    Format a feature value for clean business display.
    """
    if pd.isna(value) or value is None:
        return "missing"

    if feature in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        try:
            return f"{float(value):.3f}"
        except (ValueError, TypeError):
            return str(value)

    if "RATIO" in feature or "RATE" in feature:
        try:
            return f"{float(value):.2f}"
        except (ValueError, TypeError):
            return str(value)

    if feature in ["AGE_YEARS", "EMPLOYMENT_YEARS"]:
        try:
            return f"{float(value):.1f} years"
        except (ValueError, TypeError):
            return str(value)

    if feature in [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "TOTAL_PAYMENT_AMOUNT",
        "BUREAU_AVG_LIMIT",
        "BUREAU_TOTAL_CREDIT",
        "BUREAU_TOTAL_DEBT"
    ]:
        try:
            return f"{float(value):,.0f}"
        except (ValueError, TypeError):
            return str(value)

    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"

    return str(value)


class CreditRiskExplainer:
    """
    Explainability engine using SHAP TreeExplainer for the CatBoost model.
    """

    def __init__(
        self,
        predictor: Optional[CreditRiskPredictor] = None,
        model_path: Optional[str] = None,
        feature_columns_path: Optional[str] = None
    ) -> None:
        """
        Initialize the SHAP TreeExplainer from an existing predictor or model artifact.
        """
        if predictor is not None:
            self.predictor = predictor
        else:
            self.predictor = CreditRiskPredictor(
                model_path=model_path,
                feature_columns_path=feature_columns_path
            )

        self.model = self.predictor.model
        self.feature_columns = self.predictor.feature_columns
        self.explainer = shap.TreeExplainer(self.model)

    def explain_applicant(
        self,
        applicant_data: Union[pd.DataFrame, pd.Series, Dict[str, Any]],
        top_n: int = 5,
        user_facing_only: bool = True
    ) -> Dict[str, Any]:
        """
        Generate local SHAP explanations for an individual applicant.

        Args:
            applicant_data: Dict, Series, or single-row DataFrame.
            top_n: Number of top positive and negative features to return.
            user_facing_only: If True, excludes protected/sensitive attributes.

        Returns:
            Dictionary containing prediction summary, risk-increasing factors,
            risk-reducing factors, formatted business text, and technical SHAP data.
        """
        applicant_id, processed_df = self.predictor.preprocess_applicant(applicant_data)

        # Base prediction
        prediction = self.predictor.predict(processed_df)
        applicant_prob = prediction["Default_Probability"]

        # Calculate SHAP values for applicant
        shap_values_raw = self.explainer.shap_values(processed_df)

        # In binary classification, handle 1D or 2D SHAP output
        if isinstance(shap_values_raw, list):
            applicant_shap = shap_values_raw[1][0] if len(shap_values_raw) > 1 else shap_values_raw[0][0]
        elif len(np.array(shap_values_raw).shape) == 2:
            applicant_shap = shap_values_raw[0]
        else:
            applicant_shap = np.array(shap_values_raw).flatten()

        # Build local explanation DataFrame
        explanation_df = pd.DataFrame({
            "Feature": self.feature_columns,
            "Feature_Value": processed_df.iloc[0].values,
            "SHAP_Value": applicant_shap
        })
        explanation_df["Absolute_SHAP"] = explanation_df["SHAP_Value"].abs()

        # Optional user-facing filtering of sensitive attributes
        if user_facing_only:
            filtered_df = explanation_df[
                ~explanation_df["Feature"].isin(EXCLUDED_EXPLANATION_FEATURES)
            ].copy()
        else:
            filtered_df = explanation_df.copy()

        # Top factors increasing default risk (SHAP > 0)
        risk_increasing_df = (
            filtered_df[filtered_df["SHAP_Value"] > 0]
            .sort_values("SHAP_Value", ascending=False)
            .head(top_n)
        )

        # Top factors reducing default risk (SHAP < 0)
        risk_reducing_df = (
            filtered_df[filtered_df["SHAP_Value"] < 0]
            .sort_values("SHAP_Value", ascending=True)
            .head(top_n)
        )

        # Format business-readable sentences (strictly non-causal language)
        increasing_reasons = []
        for _, row in risk_increasing_df.iterrows():
            feat = row["Feature"]
            val = row["Feature_Value"]
            desc = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ").title())
            f_val = format_feature_value(feat, val)
            increasing_reasons.append({
                "feature": feat,
                "description": desc,
                "raw_value": val,
                "formatted_value": f_val,
                "shap_value": float(row["SHAP_Value"]),
                "readable_text": f"{desc} ({f_val}) contributed to higher predicted risk."
            })

        reducing_reasons = []
        for _, row in risk_reducing_df.iterrows():
            feat = row["Feature"]
            val = row["Feature_Value"]
            desc = FEATURE_DESCRIPTIONS.get(feat, feat.replace("_", " ").title())
            f_val = format_feature_value(feat, val)
            reducing_reasons.append({
                "feature": feat,
                "description": desc,
                "raw_value": val,
                "formatted_value": f_val,
                "shap_value": float(row["SHAP_Value"]),
                "readable_text": f"{desc} ({f_val}) contributed to lower predicted risk."
            })

        return {
            "Applicant_ID": applicant_id,
            "Default_Probability": applicant_prob,
            "Risk_Score": prediction["Risk_Score"],
            "Risk_Band": prediction["Risk_Band"],
            "Decision": prediction["Decision"],
            "Risk_Increasing_Factors": increasing_reasons,
            "Risk_Reducing_Factors": reducing_reasons,
            "User_Facing_Only": user_facing_only,
            "Base_Value": float(self.explainer.expected_value) if hasattr(self.explainer, "expected_value") else 0.0
        }
