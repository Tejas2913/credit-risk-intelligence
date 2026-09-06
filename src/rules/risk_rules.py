"""
Business Decision Rules Module
==============================
Defines the business logic for mapping default probabilities to risk bands
(Low, Medium, High) and generating recommendation rules (Approve, Manual Review, Reject).

Configuration is dynamically loaded from artifacts/risk_config.json.
"""

import json
import os
from typing import Dict, Any, Optional

DEFAULT_RISK_CONFIG_PATH = os.path.join("artifacts", "risk_config.json")


def load_risk_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the risk configuration from JSON.

    Args:
        config_path: Path to risk_config.json. Defaults to artifacts/risk_config.json.

    Returns:
        Dict containing classification_threshold, risk_bands, and decisions.
    """
    path = config_path or DEFAULT_RISK_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Risk configuration file not found at: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assign_risk_band(probability: float, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Assign a risk band (Low, Medium, High) based on default probability.

    Args:
        probability: Default probability between 0.0 and 1.0.
        config: Optional loaded risk configuration dict.

    Returns:
        Risk band string ("Low", "Medium", or "High").
    """
    if not (0.0 <= probability <= 1.0):
        # Clamp or handle edge cases gracefully
        probability = max(0.0, min(1.0, float(probability)))

    cfg = config or load_risk_config()
    risk_bands = cfg.get("risk_bands", {})

    # Default fallback boundaries if config is missing specific fields
    low_max = risk_bands.get("Low", {}).get("max_probability", 0.30)
    med_max = risk_bands.get("Medium", {}).get("max_probability", 0.60)

    if probability < low_max:
        return "Low"
    elif probability < med_max:
        return "Medium"
    else:
        return "High"


def assign_decision(risk_band: str, config: Optional[Dict[str, Any]] = None) -> str:
    """
    Map a risk band to a standardized business decision support recommendation.

    Args:
        risk_band: Risk band string ("Low", "Medium", or "High").
        config: Optional loaded risk configuration dict.

    Returns:
        Decision recommendation string.
    """
    cfg = config or load_risk_config()
    decisions = cfg.get("decisions", {})

    default_decisions = {
        "Low": "Approve / Low Risk",
        "Medium": "Manual Review",
        "High": "High Risk / Reject Recommendation"
    }

    return decisions.get(risk_band, default_decisions.get(risk_band, "Manual Review"))


def evaluate_risk_profile(probability: float, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Evaluate the full risk profile and decision recommendations for a given default probability.

    Args:
        probability: Default probability (0.0 - 1.0).
        config: Optional loaded risk configuration dict.

    Returns:
        Dictionary containing Risk_Score, Risk_Band, Predicted_Default, and Decision.
    """
    cfg = config or load_risk_config()
    threshold = float(cfg.get("classification_threshold", 0.66))

    risk_score = round(probability * 100, 2)
    risk_band = assign_risk_band(probability, cfg)
    predicted_default = int(probability >= threshold)
    decision = assign_decision(risk_band, cfg)

    return {
        "Default_Probability": float(probability),
        "Risk_Score": risk_score,
        "Risk_Band": risk_band,
        "Classification_Threshold": threshold,
        "Predicted_Default": predicted_default,
        "Decision": decision
    }
