"""
Build Risk Scores Database Module
===================================
Populates the `applicant_risk_scores` table in the analytics database
using the exact trained CatBoost model, feature store, feature order,
risk configuration, and thresholds already used by the ML API.

IMPORTANT:
- Uses the EXACT same CatBoostClassifier model loaded by CreditRiskPredictor.
- Uses the EXACT same 231 feature columns in the EXACT same order.
- Uses the EXACT same risk band thresholds from artifacts/risk_config.json.
- NEVER uses the 'target' column as a substitute for predicted risk.
- Only reads from the feature store (read-only). Only writes to analytics DB.

Usage:
    python -m src.talk_to_data.build_risk_scores
    python -m src.talk_to_data.build_risk_scores --batch-size 5000 --limit 1000
"""

import argparse
import logging
import os
import pickle
import sqlite3
import sys
import time
from typing import List, Optional

import numpy as np
import pandas as pd

# Use project root imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from catboost import CatBoostClassifier
from src.rules.risk_rules import assign_risk_band, load_risk_config
from src.talk_to_data.database import DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

DEFAULT_FEATURE_STORE_PATH = os.path.join("data", "model_features.db")
DEFAULT_MODEL_PATH = os.path.join("artifacts", "catboost_credit_risk_model.cbm")
DEFAULT_FEATURE_COLUMNS_PATH = os.path.join("artifacts", "feature_columns.pkl")
DEFAULT_RISK_CONFIG_PATH = os.path.join("artifacts", "risk_config.json")


def load_model_artifacts(
    model_path: Optional[str] = None,
    feature_columns_path: Optional[str] = None,
    risk_config_path: Optional[str] = None,
):
    """
    Load the CatBoost model, feature column order, and risk configuration.
    These are the EXACT same artifacts used by CreditRiskPredictor.
    """
    m_path = model_path or DEFAULT_MODEL_PATH
    f_path = feature_columns_path or DEFAULT_FEATURE_COLUMNS_PATH
    r_path = risk_config_path or DEFAULT_RISK_CONFIG_PATH

    if not os.path.exists(m_path):
        raise FileNotFoundError(f"Model file not found at: {m_path}")
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"Feature columns file not found at: {f_path}")
    if not os.path.exists(r_path):
        raise FileNotFoundError(f"Risk config not found at: {r_path}")

    model = CatBoostClassifier()
    model.load_model(m_path)
    logger.info("Loaded CatBoost model from: %s", m_path)

    with open(f_path, "rb") as f:
        feature_columns: List[str] = pickle.load(f)
    logger.info("Loaded %d feature columns.", len(feature_columns))

    risk_config = load_risk_config(r_path)
    logger.info("Loaded risk config from: %s", r_path)

    # Identify categorical feature names from CatBoost model
    cat_indices = model.get_cat_feature_indices()
    categorical_features = [
        feature_columns[idx] for idx in cat_indices if idx < len(feature_columns)
    ]

    return model, feature_columns, risk_config, categorical_features


def get_all_applicant_ids(feature_store_path: str) -> List[int]:
    """Retrieve all SK_ID_CURR values from the feature store."""
    conn = sqlite3.connect(f"file:{feature_store_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT SK_ID_CURR FROM applicant_model_features ORDER BY SK_ID_CURR;",
            conn
        )
        return df["SK_ID_CURR"].tolist()
    finally:
        conn.close()


def fetch_feature_batch(
    applicant_ids: List[int],
    feature_store_path: str,
    feature_columns: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """
    Fetch a batch of applicants' features from the feature store and prepare
    for CatBoost inference: exact column order, categorical fill.
    """
    placeholders = ",".join("?" * len(applicant_ids))
    conn = sqlite3.connect(f"file:{feature_store_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM applicant_model_features WHERE SK_ID_CURR IN ({placeholders});",
            conn,
            params=applicant_ids,
        )
    finally:
        conn.close()

    if df.empty:
        return df

    # Store the ID mapping before reindexing
    ids = df["SK_ID_CURR"].tolist()

    # Reindex to EXACT 231 feature columns in the EXACT order used during training
    features_df = df.reindex(columns=feature_columns)

    # Fill categorical columns with "Missing" (same as CreditRiskPredictor.preprocess_applicant)
    for col in categorical_features:
        if col in features_df.columns:
            features_df[col] = features_df[col].fillna("Missing").astype(str)

    features_df["_sk_id_curr"] = ids  # carry ID through
    return features_df


def build_risk_scores(
    db_path: Optional[str] = None,
    feature_store_path: Optional[str] = None,
    model_path: Optional[str] = None,
    feature_columns_path: Optional[str] = None,
    risk_config_path: Optional[str] = None,
    batch_size: int = 5000,
    limit: Optional[int] = None,
) -> int:
    """
    Populate the applicant_risk_scores table using CatBoost model predictions.

    Args:
        db_path: Target analytics database path.
        feature_store_path: Source feature store database path.
        model_path: CatBoost model file path.
        feature_columns_path: Pickle file with feature column order.
        risk_config_path: JSON file with risk band configuration.
        batch_size: Number of applicants per inference batch.
        limit: Optional maximum applicants to score (for testing).

    Returns:
        Number of risk score records written.
    """
    target_db = db_path or DEFAULT_DB_PATH
    feat_store = feature_store_path or DEFAULT_FEATURE_STORE_PATH

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=" * 70)
    print("BUILDING APPLICANT RISK SCORES TABLE")
    print("Source: CatBoost model predict_proba (NOT target column)")
    print("=" * 70)
    print(f"Feature Store : {feat_store}")
    print(f"Analytics DB  : {target_db}")
    print(f"Batch Size    : {batch_size:,}")
    if limit:
        print(f"Limit         : {limit:,} applicants")

    # --- Load model artifacts ---
    model, feature_columns, risk_config, categorical_features = load_model_artifacts(
        model_path=model_path,
        feature_columns_path=feature_columns_path,
        risk_config_path=risk_config_path,
    )

    # --- Get all applicant IDs from feature store ---
    all_ids = get_all_applicant_ids(feat_store)
    if limit:
        all_ids = all_ids[:limit]
    total = len(all_ids)
    print(f"\nTotal applicants to score: {total:,}")

    # --- Create/clear the risk scores table in analytics DB ---
    conn_write = sqlite3.connect(target_db)
    try:
        conn_write.execute("""
            CREATE TABLE IF NOT EXISTS applicant_risk_scores (
                sk_id_curr          INTEGER PRIMARY KEY,
                default_probability REAL NOT NULL,
                risk_score          REAL NOT NULL,
                risk_band           TEXT NOT NULL
            )
        """)
        conn_write.execute("DELETE FROM applicant_risk_scores;")
        conn_write.execute("CREATE INDEX IF NOT EXISTS idx_risk_band ON applicant_risk_scores(risk_band);")
        conn_write.execute("CREATE INDEX IF NOT EXISTS idx_default_probability ON applicant_risk_scores(default_probability);")
        conn_write.execute("CREATE INDEX IF NOT EXISTS idx_risk_score ON applicant_risk_scores(risk_score);")
        conn_write.commit()
        print("Table applicant_risk_scores ready (cleared for rebuild).")
    finally:
        conn_write.close()

    # --- Batch inference ---
    total_written = 0
    start_time = time.time()

    for batch_start in range(0, total, batch_size):
        batch_ids = all_ids[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        features_df = fetch_feature_batch(
            applicant_ids=batch_ids,
            feature_store_path=feat_store,
            feature_columns=feature_columns,
            categorical_features=categorical_features,
        )

        if features_df.empty:
            logger.warning("Batch %d/%d: no features returned.", batch_num, total_batches)
            continue

        # Extract IDs before dropping
        sk_ids = features_df["_sk_id_curr"].tolist()
        features_only = features_df.drop(columns=["_sk_id_curr"])

        # CatBoost predict_proba — class 1 = default
        proba_array = model.predict_proba(features_only)
        default_probs = proba_array[:, 1]  # column 1 = P(default)

        # Apply risk banding using exact same config as CreditRiskPredictor
        records = []
        for sk_id, prob in zip(sk_ids, default_probs):
            prob_f = float(prob)
            risk_score = round(prob_f * 100, 4)
            risk_band = assign_risk_band(prob_f, risk_config)
            records.append((int(sk_id), round(prob_f, 6), risk_score, risk_band))

        # Write batch to analytics DB
        conn_write = sqlite3.connect(target_db)
        try:
            conn_write.executemany(
                "INSERT OR REPLACE INTO applicant_risk_scores "
                "(sk_id_curr, default_probability, risk_score, risk_band) "
                "VALUES (?, ?, ?, ?);",
                records,
            )
            conn_write.commit()
        finally:
            conn_write.close()

        total_written += len(records)
        elapsed = time.time() - start_time
        rate = total_written / elapsed if elapsed > 0 else 0
        eta_s = (total - total_written) / rate if rate > 0 else 0

        print(
            f"  Batch {batch_num}/{total_batches}: "
            f"{total_written:,}/{total:,} scored "
            f"({total_written * 100 / total:.1f}%) | "
            f"{rate:.0f}/s | ETA: {eta_s:.0f}s"
        )

    elapsed_total = time.time() - start_time
    print(f"\nCompleted: {total_written:,} risk scores written in {elapsed_total:.1f}s.")

    # --- Verify ---
    conn_verify = sqlite3.connect(f"file:{target_db}?mode=ro", uri=True)
    try:
        df_check = pd.read_sql_query(
            "SELECT risk_band, COUNT(*) AS count, "
            "ROUND(AVG(default_probability) * 100, 2) AS avg_prob_pct "
            "FROM applicant_risk_scores GROUP BY risk_band ORDER BY avg_prob_pct DESC;",
            conn_verify,
        )
        print("\nRisk Band Distribution (from model predictions):")
        print(df_check.to_string(index=False))
    finally:
        conn_verify.close()

    return total_written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build applicant_risk_scores from CatBoost model.")
    parser.add_argument("--batch-size", type=int, default=5000, help="Batch size for inference.")
    parser.add_argument("--limit", type=int, default=None, help="Max applicants to score (for testing).")
    parser.add_argument("--db-path", type=str, default=None, help="Target analytics database path.")
    args = parser.parse_args()

    count = build_risk_scores(
        db_path=args.db_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    sys.exit(0 if count > 0 else 1)
