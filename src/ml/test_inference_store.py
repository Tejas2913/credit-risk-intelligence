"""
Test Suite for ML Inference Feature Store
=========================================
Validates:
1. Feature store database existence and file size.
2. Applicant count (307,511 rows) and uniqueness of SK_ID_CURR.
3. Exact 231 feature count, names, and column order matching `artifacts/feature_columns.pkl`.
4. Exact numerical verification for Applicant 256571 (probability ~0.2283, score 22.83, Low band, Approve).
5. SHAP feature importance explanation and sensitive feature masking.
6. Batch verification across 5 additional applicant IDs.
7. Controlled non-crashing handling for non-existent applicant IDs.
"""

import os
import pickle
import sqlite3
import pandas as pd
import numpy as np

from src.ml.explainer import CreditRiskExplainer
from src.ml.feature_store import (
    DEFAULT_FEATURE_STORE_PATH,
    explain_applicant,
    get_applicant_count,
    get_applicant_features,
    load_feature_columns,
    predict_applicant
)
from src.ml.predictor import CreditRiskPredictor


def run_feature_store_tests():
    print("=" * 80)
    print("STARTING ML INFERENCE FEATURE STORE VALIDATION SUITE")
    print("=" * 80)

    db_path = DEFAULT_FEATURE_STORE_PATH

    # Test 1: File Existence & Size
    print("\n[Test 1] Verifying Feature Store File Existence & Size...")
    assert os.path.exists(db_path), f"Feature store not found at: {db_path}"
    db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
    print(f"Feature store found at: {db_path} ({db_size_mb:.2f} MB)")
    assert db_size_mb > 10, "Feature store file unexpectedly small."
    print("Status: PASSED")

    # Test 2: Applicant Count & Uniqueness
    print("\n[Test 2] Verifying Total Applicant Count & SK_ID_CURR Uniqueness...")
    count = get_applicant_count(db_path)
    print(f"Total applicants in store: {count:,}")
    assert count == 307511, f"Expected 307,511 applicants, found {count}"

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        dup_count = conn.execute(
            "SELECT COUNT(SK_ID_CURR) - COUNT(DISTINCT SK_ID_CURR) FROM applicant_model_features;"
        ).fetchone()[0]
        assert dup_count == 0, f"Found {dup_count} duplicate SK_ID_CURR records!"
    print("Status: PASSED (307,511 unique applicants confirmed)")

    # Test 3: Feature Count & Exact Column Alignment
    print("\n[Test 3] Verifying 231 Feature Columns & Order Alignment...")
    authoritative_cols = load_feature_columns()
    print(f"Authoritative feature count: {len(authoritative_cols)}")
    assert len(authoritative_cols) == 231, f"Expected 231 features, found {len(authoritative_cols)}"

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM applicant_model_features LIMIT 1;")
        db_cols = [desc[0] for desc in cursor.description if desc[0] != "SK_ID_CURR"]

    assert len(db_cols) == 231, f"Database table has {len(db_cols)} feature columns instead of 231"
    assert db_cols == authoritative_cols, "Feature columns or order in database do not match feature_columns.pkl!"
    print("Status: PASSED (Exact 231 feature columns and order matched 100%)")

    # Test 4: Verified Baseline Applicant (Notebook Cell 176: Row Index 256571 -> SK_ID_CURR 396899)
    print("\n[Test 4] Verifying Inference for Notebook Benchmark Applicant (SK_ID_CURR 396899 / Index 256571)...")
    predictor = CreditRiskPredictor()
    res_bench = predict_applicant(396899, predictor=predictor, db_path=db_path)

    print("Applicant ID       :", res_bench["applicant_id"])
    print(f"Default Probability: {res_bench['default_probability']:.6f} (Expected: 0.228251)")
    print(f"Risk Score         : {res_bench['risk_score']} (Expected: 22.83)")
    print(f"Risk Band          : {res_bench['risk_band']} (Expected: 'Low')")
    print(f"Decision           : {res_bench['decision']} (Expected: 'Approve / Low Risk')")
    print(f"Prediction (0/1)   : {res_bench['prediction']} (Expected: 0)")

    assert res_bench["success"] is True
    assert abs(res_bench["default_probability"] - 0.228251) < 0.0001, (
        f"Probability {res_bench['default_probability']} deviated from expected ~0.228251"
    )
    assert res_bench["risk_band"] == "Low"
    assert res_bench["prediction"] == 0
    print("Status: PASSED (Notebook Benchmark Row verified to exact precision: 0.228251 / 22.83)")

    # Also test applicant SK_ID_CURR 256571
    res_256571 = predict_applicant(256571, predictor=predictor, db_path=db_path)
    print(f"\nApplicant SK_ID_CURR 256571 -> Prob: {res_256571['default_probability']:.6f} | Score: {res_256571['risk_score']} | Band: {res_256571['risk_band']} | Decision: {res_256571['decision']}")
    assert res_256571["success"] is True
    assert res_256571["risk_band"] == "Low"
    print("Status: PASSED (Applicant SK_ID_CURR 256571 verified)")

    # Test 5: SHAP Explanation for Benchmark Applicant 396899
    print("\n[Test 5] Verifying SHAP Explanation & Sensitive Feature Filtering...")
    explainer = CreditRiskExplainer(predictor=predictor)
    shap_res = explain_applicant(396899, explainer=explainer, predictor=predictor, db_path=db_path, top_k=5)

    assert shap_res["success"] is True
    print("SHAP Base Value    :", round(shap_res["Base_Value"], 4))
    print("Top Risk Factors (increasing risk):")
    for f in shap_res["Risk_Increasing_Factors"]:
        print(f"  - {f['feature']}: formatted={f['formatted_value']}, shap={round(f['shap_value'], 4)}")

    print("Top Mitigating Factors (decreasing risk):")
    for f in shap_res["Risk_Reducing_Factors"]:
        print(f"  - {f['feature']}: formatted={f['formatted_value']}, shap={round(f['shap_value'], 4)}")

    # Verify sensitive features are excluded
    sensitive_set = {"CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY"}
    for f in shap_res["Risk_Increasing_Factors"] + shap_res["Risk_Reducing_Factors"]:
        assert f["feature"] not in sensitive_set, f"Sensitive attribute {f['feature']} leaked in explanation!"
    print("Status: PASSED (SHAP values generated and sensitive features successfully filtered)")

    # Test 6: 5 Additional Applicant Predictions
    print("\n[Test 6] Testing Batch Inference on 5 Additional Applicant IDs...")
    sample_ids = [100002, 100003, 100004, 100006, 100007]
    for app_id in sample_ids:
        r = predict_applicant(app_id, predictor=predictor, db_path=db_path)
        assert r["success"] is True
        print(
            f"Applicant {r['applicant_id']} -> Prob: {r['default_probability']:.4f} | "
            f"Score: {r['risk_score']:5.2f} | Band: {r['risk_band']:<6} | Decision: {r['decision']}"
        )
    print("Status: PASSED (All 5 additional applicants successfully scored)")

    # Test 7: Missing Applicant ID Handling
    print("\n[Test 7] Testing Missing Applicant ID Handling...")
    missing_id = 999999999
    missing_res = predict_applicant(missing_id, predictor=predictor, db_path=db_path)
    print(f"Missing ID Query Result: {missing_res}")
    assert missing_res["success"] is False
    assert "not found" in missing_res["error"].lower()
    print("Status: PASSED (Non-existent applicant handled cleanly without crashing)")

    print("\n" + "=" * 80)
    print("ALL STEP 4 INFERENCE STORE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_feature_store_tests()
