"""
Comprehensive FastAPI Backend Test Suite
========================================
Validates all endpoints across the 3 core platform modules:
1. Health & Status (/health)
2. EDA / Analytics (/api/eda/summary, /api/eda/insights)
3. ML Inference & XAI (/api/ml/applicant/{id}, /api/ml/applicant/{id}/explanation, 404 handling)
4. Talk-to-Data Chatbot (/api/chat, invalid payloads, underwriting redirects)
5. Runtime Raw CSV Independence & Security Validation
"""

import math
import os
import unittest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)

BENCHMARK_APPLICANT_ID = 396899
NONEXISTENT_APPLICANT_ID = 999999999


class TestAPIBackend(unittest.TestCase):
    """Integration and unit tests for the Unified FastAPI backend."""

    # ------------------------------------------------------------------
    # Health & System Status Tests
    # ------------------------------------------------------------------

    def test_01_health_endpoint(self):
        """Test GET /health returns 200 OK with all subsystems healthy."""
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "credit-risk-intelligence-api")
        self.assertEqual(data["version"], "1.0.0")
        self.assertTrue(data["databases"]["analytics_db"])
        self.assertTrue(data["databases"]["feature_store_db"])
        self.assertTrue(data["model_loaded"])

    def test_02_root_endpoint(self):
        """Test GET / returns welcome JSON and documentation links."""
        response = client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("documentation", data)
        self.assertEqual(data["documentation"], "/docs")

    # ------------------------------------------------------------------
    # Module 1: EDA / Analytics Tests
    # ------------------------------------------------------------------

    def test_03_eda_summary_endpoint(self):
        """Test GET /api/eda/summary returns accurate dataset totals from SQLite analytics DB."""
        response = client.get("/api/eda/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertEqual(data["total_applicants"], 307511)
        self.assertEqual(data["default_applicants"], 24825)
        self.assertEqual(data["non_default_applicants"], 282686)
        self.assertAlmostEqual(data["default_rate"], 0.080729, places=5)
        self.assertAlmostEqual(data["default_rate_pct"], 8.07, places=1)
        self.assertEqual(data["total_features"], 231)
        self.assertEqual(data["numerical_features"], 106)
        self.assertEqual(data["categorical_features"], 16)

    def test_04_eda_insights_endpoint(self):
        """Test GET /api/eda/insights returns all 6 core business insights with chart data."""
        response = client.get("/api/eda/insights")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("insights", data)
        insights = data["insights"]
        self.assertEqual(len(insights), 6)

        # Verify structure of each insight
        expected_titles = [
            "Distribution of Loan Default Outcomes",
            "Default Rate by Previous Application Outcome",
            "Default Rate by Historical POS/CASH Delinquency",
            "Default Rate by Credit Card Utilization",
            "Default Rate by Historical Installment Payment Behavior",
            "Default Rate by Historical Late-Payment Frequency"
        ]

        for i, insight in enumerate(insights):
            self.assertEqual(insight["id"], i + 1)
            self.assertEqual(insight["title"], expected_titles[i])
            self.assertIn("metric", insight)
            self.assertIn("interpretation", insight)
            self.assertIn("chart", insight)
            chart = insight["chart"]
            self.assertIsInstance(chart["labels"], list)
            self.assertIsInstance(chart["values"], list)
            self.assertEqual(len(chart["labels"]), len(chart["values"]))
            self.assertGreater(len(chart["labels"]), 0)

    # ------------------------------------------------------------------
    # Module 2: ML Inference & SHAP/XAI Tests
    # ------------------------------------------------------------------

    def test_05_ml_applicant_prediction_benchmark(self):
        """Test GET /api/ml/applicant/396899 reproduces the exact notebook benchmark."""
        response = client.get(f"/api/ml/applicant/{BENCHMARK_APPLICANT_ID}")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["applicant_id"], BENCHMARK_APPLICANT_ID)

        # Check probability: ~0.228251 (tolerance 0.005)
        self.assertAlmostEqual(data["default_probability"], 0.228251, places=3)
        self.assertAlmostEqual(data["risk_score"], 22.83, places=1)
        self.assertEqual(data["risk_band"], "Low")
        self.assertEqual(data["prediction"], 0)
        self.assertIn("Approve", data["decision"])

    def test_06_ml_applicant_shap_explanation(self):
        """Test GET /api/ml/applicant/396899/explanation returns factors and masks sensitive attributes."""
        response = client.get(f"/api/ml/applicant/{BENCHMARK_APPLICANT_ID}/explanation?top_k=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(data["applicant_id"], BENCHMARK_APPLICANT_ID)
        self.assertIn("base_value", data)
        self.assertIn("risk_increasing_factors", data)
        self.assertIn("risk_reducing_factors", data)
        self.assertIn("disclaimer", data)

        sensitive_attrs = {
            "CODE_GENDER", "NAME_FAMILY_STATUS", "NAME_EDUCATION_TYPE",
            "FLAG_OWN_CAR", "FLAG_OWN_REALTY"
        }

        all_factors = data["risk_increasing_factors"] + data["risk_reducing_factors"]
        self.assertGreater(len(all_factors), 0)

        for factor in all_factors:
            self.assertNotIn(
                factor["feature"].upper(),
                sensitive_attrs,
                f"Sensitive attribute {factor['feature']} was exposed in user-facing explanation!"
            )
            self.assertIn("shap_value", factor)
            self.assertIn("direction", factor)

    def test_07_ml_applicant_404_nonexistent(self):
        """Test GET /api/ml/applicant/999999999 returns controlled HTTP 404 error."""
        response = client.get(f"/api/ml/applicant/{NONEXISTENT_APPLICANT_ID}")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())

    def test_08_ml_explanation_404_nonexistent(self):
        """Test GET /api/ml/applicant/999999999/explanation returns controlled HTTP 404 error."""
        response = client.get(f"/api/ml/applicant/{NONEXISTENT_APPLICANT_ID}/explanation")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("not found", data["detail"].lower())

    # ------------------------------------------------------------------
    # Module 3: Talk-to-Data Chat Tests
    # ------------------------------------------------------------------

    def test_09_chat_endpoint_default_rate_query(self):
        """Test POST /api/chat with 'What is the default rate?' returns accurate analytics."""
        payload = {"message": "What is the default rate?"}
        response = client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertIn("8.07", data["answer"])
        self.assertIsNotNone(data["sql"])
        self.assertTrue(len(data.get("intent", "")) > 0)
        self.assertGreaterEqual(len(data["data"]), 1)

    def test_10_chat_endpoint_segmented_query(self):
        """Test POST /api/chat with 'Show default rate by education level'."""
        payload = {"message": "Show default rate by education level", "session_id": "test_session_1"}
        response = client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertIsNotNone(data["sql"])
        self.assertIn("NAME_EDUCATION_TYPE", data["sql"].upper())
        self.assertGreater(len(data["data"]), 1)

    def test_11_chat_underwriting_redirect(self):
        """Test POST /api/chat with individual underwriting query redirects to ML predictor."""
        payload = {"message": "Should we approve loan application 396899?"}
        response = client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertFalse(data["success"])
        self.assertIn("redirect", data["intent"])
        self.assertIn("Credit Risk", data["answer"])
        self.assertIsNone(data["sql"])

    def test_12_chat_invalid_payload(self):
        """Test POST /api/chat with empty message returns 422 Unprocessable Entity."""
        response = client.post("/api/chat", json={"message": "   "})
        self.assertEqual(response.status_code, 422)

    def test_14_chat_top_5_high_risk_applicants(self):
        """Test POST /api/chat with 'Show me the top 5 high-risk applicants' returns 5 rows and aligned narrative."""
        payload = {"message": "Show me the top 5 high-risk applicants", "session_id": "test_session_top5"}
        response = client.post("/api/chat", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertTrue(data["success"])
        self.assertEqual(len(data["data"]), 5)
        self.assertIsNotNone(data["sql"])
        self.assertIn("applicant_risk_scores", data["sql"].lower())
        
        # Verify narrative alignment
        answer = data["answer"]
        self.assertIn("highest", answer.lower())
        self.assertIn("default risk", answer.lower())
        self.assertNotIn("None applicants", answer)
        self.assertNotIn("Secondary / secondary special", answer)
        self.assertNotIn("at None", answer)

        # Verify semantic variations
        variations = [
            "Give me the five riskiest applicants",
            "Show the highest-risk 5 applicants",
            "Who are the top 5 applicants by predicted default risk?",
            "List five applicants with the highest risk scores"
        ]
        for q in variations:
            res = client.post("/api/chat", json={"message": q, "session_id": f"test_var_{q[:10]}"})
            self.assertEqual(res.status_code, 200)
            res_data = res.json()
            self.assertTrue(res_data["success"], f"Failed on semantic variation: {q}")
            self.assertEqual(len(res_data["data"]), 5, f"Expected 5 rows for variation: {q}")
            self.assertIn("highest", res_data["answer"].lower())
            self.assertNotIn("None applicants", res_data["answer"])

    # ------------------------------------------------------------------
    # Data Independence & Safety Tests
    # ------------------------------------------------------------------

    def test_15_no_raw_csv_dependency(self):
        """Verify that the repository contains NO raw CSV files at runtime."""
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        raw_csv_names = [
            "application_train.csv", "bureau.csv", "bureau_balance.csv",
            "previous_application.csv", "POS_CASH_balance.csv",
            "credit_card_balance.csv", "installments_payments.csv"
        ]

        found_csvs = []
        for root, _, files in os.walk(repo_root):
            for file in files:
                if file in raw_csv_names:
                    found_csvs.append(os.path.join(root, file))

        self.assertEqual(
            found_csvs, [],
            f"Raw CSV files found in repository! {found_csvs}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

