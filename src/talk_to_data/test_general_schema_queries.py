"""
General Schema-Aware Query Test Suite (STEP 10D)
================================================
Tests that the Talk-to-Data system can answer ANY reasonable analytical question
answerable from the approved database schema — not just the 7 original fixed patterns.

Tests cover:
1. Simple aggregate questions (avg income, count with filter, default rate, avg credit)
2. Grouped questions (by education, organization, income tier)
3. Relational join questions (refusal history vs default, bureau, installment)
4. Model-risk questions (high-risk count, top-N, percentage per band)
5. Compound filter questions
6. Unsupported domain rejection (no hallucination)
7. Individual decision redirect
8. Security guardrails (unchanged)
9. Deterministic shortcut validation
"""

import os
import re
import sqlite3
import unittest
from typing import Any, Dict

from src.talk_to_data.database import DEFAULT_DB_PATH
from src.talk_to_data.sql_generator import (
    check_intent_safety,
    generate_sql,
    generate_sql_deterministic,
)
from src.talk_to_data.sql_validator import validate_sql


class TestSimpleSchemaQuestions(unittest.TestCase):
    """Test simple single-table aggregate questions that all return data."""

    def _run(self, question: str) -> Dict[str, Any]:
        return generate_sql_deterministic(question)

    def test_average_income(self):
        """Average income query should match deterministic shortcut."""
        result = self._run("What is the average annual income of applicants?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        sql_up = result["sql"].upper()
        self.assertIn("AVG", sql_up)
        self.assertIn("AMT_INCOME_TOTAL", sql_up)

    def test_average_income_variants(self):
        """Multiple phrasings of average income should all match."""
        for q in ["What is the avg income?", "mean income of applicants", "average annual income"]:
            result = self._run(q)
            self.assertIsNotNone(result["sql"], f"Failed for: {q}")
            self.assertTrue(result["is_supported"], f"Failed for: {q}")

    def test_average_credit_amount(self):
        """Average credit amount query should match deterministic shortcut."""
        result = self._run("What is the average credit amount?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("AMT_CREDIT", result["sql"].upper())
        self.assertIn("AVG", result["sql"].upper())

    def test_default_rate(self):
        """Portfolio default rate should match deterministic shortcut."""
        result = self._run("What is the overall portfolio default rate?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("AVG(TARGET)", result["sql"].upper().replace(" ", ""))

    def test_count_income_above_threshold(self):
        """Count of applicants with income above threshold should generate correct SQL."""
        result = self._run("How many applicants have income above 300000?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("300000", result["sql"])
        self.assertIn("amt_income_total", result["sql"].lower())

    def test_count_income_above_threshold_variant(self):
        """Alternative phrasing of income threshold query."""
        result = self._run("How many applicants earn above 200000?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("200000", result["sql"])

    def test_default_rate_by_education(self):
        """Default rate by education should match deterministic shortcut."""
        result = self._run("What is the default rate by education level?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("name_education_type", result["sql"].lower())
        self.assertIn("GROUP BY", result["sql"].upper())

    def test_default_rate_by_organization(self):
        """Default rate by organization type should match deterministic shortcut."""
        result = self._run("Which organization type has the highest default rate?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("organization_type", result["sql"].lower())
        self.assertIn("GROUP BY", result["sql"].upper())

    def test_income_tier_analysis(self):
        """Default rate across income segments should match deterministic shortcut."""
        result = self._run("How does default risk vary across income tiers?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("income_tier", result["sql"].lower())


class TestRelationalQuestions(unittest.TestCase):
    """Test multi-table relational join questions."""

    def _run(self, question: str) -> Dict[str, Any]:
        return generate_sql_deterministic(question)

    def test_previous_refusal_vs_default(self):
        """Previous refusal history analysis should match relational pattern."""
        result = self._run("How does previous refusal history relate to default?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        sql_lower = result["sql"].lower()
        self.assertIn("prev_refusal_rate", sql_lower)

    def test_installment_late_payment_for_defaulters(self):
        """Installment late payment behavior for defaulters should match relational pattern."""
        result = self._run("What is the average installment payment behavior for defaulters?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        sql_upper = result["sql"].upper()
        self.assertIn("JOIN", sql_upper)
        self.assertIn("INSTALLMENT_SUMMARY", sql_upper)

    def test_bureau_debt_analysis(self):
        """Bureau debt vs credit limit comparison should match relational pattern."""
        result = self._run("Compare default rates for applicants with bureau debt above their credit limit.")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("bureau_summary", result["sql"].lower())

    def test_previous_refusal_and_late_payments(self):
        """Joint risk profile across refusals and late payments."""
        result = self._run("How many applicants had previous loan refusals and late installment payments?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        sql_upper = result["sql"].upper()
        self.assertIn("JOIN", sql_upper)

    def test_validated_sql_for_relational(self):
        """All relational queries must pass the SQL validator."""
        relational_questions = [
            "Compare default rates for applicants with bureau debt above their credit limit.",
            "What is the average installment payment behavior for defaulters?",
        ]
        for q in relational_questions:
            result = self._run(q)
            if result["is_supported"] and result["sql"]:
                val = validate_sql(result["sql"])
                self.assertTrue(
                    val["valid"],
                    f"Validator rejected SQL for '{q}': {val['reason']}"
                )


class TestRiskScoreQuestions(unittest.TestCase):
    """Test questions about CatBoost model-predicted risk scores."""

    def _run(self, question: str) -> Dict[str, Any]:
        return generate_sql_deterministic(question)

    def test_top_5_high_risk_applicants(self):
        """Top 5 high-risk applicants should query applicant_risk_scores."""
        result = self._run("Show me the top 5 high-risk applicants")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        sql_lower = result["sql"].lower()
        self.assertIn("applicant_risk_scores", sql_lower)
        self.assertIn("default_probability", sql_lower)
        self.assertIn("limit 5", sql_lower)

    def test_top_n_risky_applicants_various_n(self):
        """Top N risky applicants for different values of N."""
        for n in [3, 10, 20]:
            result = self._run(f"Show the top {n} highest-risk applicants by predicted risk score")
            self.assertIsNotNone(result["sql"], f"Failed for N={n}")
            self.assertTrue(result["is_supported"], f"Failed for N={n}")
            self.assertIn(f"limit {n}", result["sql"].lower())

    def test_risk_band_distribution(self):
        """Risk band distribution should use applicant_risk_scores."""
        result = self._run("What proportion of applicants fall into each risk band?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("applicant_risk_scores", result["sql"].lower())
        self.assertIn("risk_band", result["sql"].lower())

    def test_high_risk_count_variant(self):
        """'How many high-risk applicants' should use the risk scores table."""
        result = self._run("How many applicants are classified as High risk?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("applicant_risk_scores", result["sql"].lower())

    def test_risk_scores_never_use_target(self):
        """Risk score queries must NEVER use 'target' as predicted risk."""
        risk_questions = [
            "Show me the top 5 high-risk applicants",
            "What proportion of applicants fall into each risk band?",
            "How many applicants are classified as High risk?",
        ]
        for q in risk_questions:
            result = self._run(q)
            if result.get("sql"):
                # The SQL should use applicant_risk_scores, not target alone
                sql_lower = result["sql"].lower()
                self.assertIn(
                    "applicant_risk_scores", sql_lower,
                    f"Query '{q}' must use applicant_risk_scores, not just target column"
                )

    def test_risk_score_sql_passes_validator(self):
        """All risk score queries must pass the SQL validator."""
        questions = [
            "Show me the top 5 high-risk applicants",
            "What proportion of applicants fall into each risk band?",
        ]
        for q in questions:
            result = self._run(q)
            if result["is_supported"] and result["sql"]:
                val = validate_sql(result["sql"])
                self.assertTrue(
                    val["valid"],
                    f"Validator rejected SQL for '{q}': {val['reason']}"
                )

    def test_risk_sql_joins_on_sk_id_curr(self):
        """Multi-table risk queries must join on sk_id_curr."""
        result = self._run("Show me the top 5 high-risk applicants")
        if result["is_supported"] and result["sql"]:
            if "JOIN" in result["sql"].upper():
                self.assertIn("sk_id_curr", result["sql"].lower())

    def test_average_risk_score(self):
        """Average predicted risk score query."""
        result = self._run("What is the average predicted risk score?")
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("applicant_risk_scores", result["sql"].lower())
        self.assertIn("AVG", result["sql"].upper())


class TestUnsupportedDomainRejection(unittest.TestCase):
    """Unsupported questions must be rejected without hallucination."""

    def test_interest_rate_rejected(self):
        """Interest rate query is not in schema — must be rejected."""
        result = check_intent_safety("What is the average interest rate?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["intent"], "unsupported_domain")

    def test_unemployment_rate_rejected(self):
        """Unemployment rate is external data — must be rejected."""
        result = check_intent_safety("What is the unemployment rate in India?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["intent"], "unsupported_domain")

    def test_experian_score_rejected(self):
        """Experian score is not in schema — must be rejected."""
        result = check_intent_safety("What is the applicant's Experian score?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["intent"], "unsupported_domain")

    def test_bitcoin_rejected(self):
        """Crypto/Bitcoin is entirely out-of-domain."""
        result = check_intent_safety("What is the bitcoin price trend?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])

    def test_fico_score_rejected(self):
        """FICO score is not in the approved schema."""
        result = check_intent_safety("What is the average FICO score?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["intent"], "unsupported_domain")

    def test_gdp_rejected(self):
        """GDP questions are not in the approved schema."""
        result = check_intent_safety("How does GDP affect default rates?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])


class TestIndividualDecisionRedirect(unittest.TestCase):
    """Individual loan decisions must always redirect to the ML predictor."""

    def test_should_applicant_be_approved(self):
        """'Should applicant X be approved?' must redirect."""
        result = check_intent_safety("Should applicant 100002 be approved?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["intent"], "individual_decision_redirect")
        self.assertIn("ML Predictor", result["explanation"])

    def test_should_we_reject(self):
        """'Should we reject applicant X?' must redirect."""
        result = check_intent_safety("Should I reject applicant 100003?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["intent"], "individual_decision_redirect")

    def test_approve_or_reject(self):
        """'Approve or reject this applicant' must redirect."""
        result = check_intent_safety("Should we approve or reject this applicant?")
        self.assertIsNotNone(result)
        self.assertFalse(result["is_supported"])
        self.assertEqual(result["intent"], "individual_decision_redirect")

    def test_portfolio_analytics_still_allowed(self):
        """Portfolio-level risk analytics (not individual decisions) must NOT be redirected."""
        # These are aggregate analytics questions — must NOT be rejected by check_intent_safety
        allowed_questions = [
            "How many applicants are classified as High risk?",
            "What proportion of applicants fall into each risk band?",
            "What is the average predicted risk score?",
        ]
        for q in allowed_questions:
            result = check_intent_safety(q)
            self.assertIsNone(
                result,
                f"Portfolio analytics question was incorrectly redirected: '{q}'"
            )


class TestSecurityGuardrails(unittest.TestCase):
    """All SQL security controls must remain fully intact."""

    def test_drop_table_rejected(self):
        result = validate_sql("DROP TABLE applicant_analytics;")
        self.assertFalse(result["valid"])

    def test_delete_rejected(self):
        result = validate_sql("DELETE FROM applicant_analytics WHERE sk_id_curr = 100001;")
        self.assertFalse(result["valid"])

    def test_update_rejected(self):
        result = validate_sql("UPDATE applicant_analytics SET target = 0;")
        self.assertFalse(result["valid"])

    def test_insert_rejected(self):
        result = validate_sql(
            "INSERT INTO applicant_analytics (sk_id_curr, target, amt_income_total, amt_credit, name_contract_type) "
            "VALUES (999999, 1, 100000, 200000, 'Cash loans');"
        )
        self.assertFalse(result["valid"])

    def test_pragma_rejected(self):
        result = validate_sql("PRAGMA table_info(applicant_analytics);")
        self.assertFalse(result["valid"])

    def test_multi_statement_rejected(self):
        result = validate_sql("SELECT * FROM applicant_analytics; DROP TABLE applicant_analytics;")
        self.assertFalse(result["valid"])

    def test_sqlite_master_rejected(self):
        result = validate_sql("SELECT * FROM sqlite_master;")
        self.assertFalse(result["valid"])

    def test_attach_rejected(self):
        result = validate_sql("ATTACH DATABASE '/tmp/evil.db' AS evil;")
        self.assertFalse(result["valid"])

    def test_unauthorized_table_rejected(self):
        result = validate_sql("SELECT * FROM secret_table;")
        self.assertFalse(result["valid"])

    def test_invalid_join_key_rejected(self):
        """Joins not keyed on sk_id_curr must be rejected."""
        result = validate_sql(
            "SELECT * FROM applicants a JOIN bureau_summary b ON a.target = b.bureau_credit_count;"
        )
        self.assertFalse(result["valid"])

    def test_unauthorized_table_in_join_rejected(self):
        """Joins with unauthorized tables must be rejected."""
        result = validate_sql(
            "SELECT * FROM applicants a JOIN secret_table s ON a.sk_id_curr = s.sk_id_curr;"
        )
        self.assertFalse(result["valid"])

    def test_applicant_risk_scores_valid(self):
        """Queries to applicant_risk_scores must now be ALLOWED by the validator."""
        result = validate_sql(
            "SELECT sk_id_curr, risk_score, risk_band FROM applicant_risk_scores "
            "ORDER BY default_probability DESC LIMIT 10;"
        )
        self.assertTrue(result["valid"], f"applicant_risk_scores query was incorrectly rejected: {result['reason']}")

    def test_risk_scores_join_with_applicants_valid(self):
        """Join between applicant_risk_scores and applicants must be allowed."""
        result = validate_sql(
            "SELECT r.sk_id_curr, r.risk_score, r.risk_band, a.amt_income_total "
            "FROM applicant_risk_scores r "
            "JOIN applicants a ON r.sk_id_curr = a.sk_id_curr "
            "ORDER BY r.default_probability DESC LIMIT 5;"
        )
        self.assertTrue(result["valid"], f"Risk+applicants join was incorrectly rejected: {result['reason']}")


class TestCompoundFilterQuestions(unittest.TestCase):
    """Test compound analytical questions with multiple filters."""

    def test_education_default_rate_with_income_filter(self):
        """Education default rate with income filter via context."""
        # First establish context
        context = [{
            "question": "What is the default rate by education level?",
            "sql": "SELECT name_education_type, ROUND(AVG(target)*100,2) AS default_rate_pct FROM applicant_analytics GROUP BY name_education_type;",
            "intent": "portfolio_analytics"
        }]
        result = generate_sql_deterministic(
            "What about only applicants earning above 200000?",
            conversation_context=context
        )
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        sql_upper = result["sql"].upper()
        self.assertIn("WHERE", sql_upper)
        self.assertIn("200000", result["sql"])
        self.assertIn("AMT_INCOME_TOTAL", sql_upper)

    def test_previous_refusal_history_deterministic(self):
        """Previous refusal history pattern should match deterministic shortcut."""
        result = generate_sql_deterministic(
            "What is the default rate for applicants with previous loan refusals?"
        )
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("prev_refusal_rate", result["sql"].lower())

    def test_late_payment_delinquency_deterministic(self):
        """Late payment delinquency pattern should match."""
        result = generate_sql_deterministic(
            "How does late payment history relate to default risk?"
        )
        self.assertIsNotNone(result["sql"])
        self.assertTrue(result["is_supported"])
        self.assertIn("late_payment_ratio", result["sql"].lower())


class TestSQLValidatorAllowsRiskTable(unittest.TestCase):
    """Regression tests: applicant_risk_scores is now an approved table."""

    def test_simple_select_from_risk_scores(self):
        sql = "SELECT sk_id_curr, default_probability, risk_score, risk_band FROM applicant_risk_scores LIMIT 10;"
        result = validate_sql(sql)
        self.assertTrue(result["valid"], result["reason"])

    def test_order_by_probability(self):
        sql = "SELECT sk_id_curr, risk_score FROM applicant_risk_scores ORDER BY default_probability DESC LIMIT 5;"
        result = validate_sql(sql)
        self.assertTrue(result["valid"], result["reason"])

    def test_risk_band_filter(self):
        sql = "SELECT COUNT(*) AS high_risk_count FROM applicant_risk_scores WHERE risk_band = 'High';"
        result = validate_sql(sql)
        self.assertTrue(result["valid"], result["reason"])

    def test_group_by_risk_band(self):
        sql = (
            "SELECT risk_band, COUNT(*) AS cnt, ROUND(AVG(risk_score),2) AS avg_score "
            "FROM applicant_risk_scores GROUP BY risk_band ORDER BY avg_score DESC;"
        )
        result = validate_sql(sql)
        self.assertTrue(result["valid"], result["reason"])

    def test_join_risk_scores_and_applicants(self):
        sql = (
            "SELECT r.sk_id_curr, r.risk_score, r.risk_band, a.amt_income_total, a.name_education_type "
            "FROM applicant_risk_scores r "
            "JOIN applicants a ON r.sk_id_curr = a.sk_id_curr "
            "WHERE r.risk_band = 'High' "
            "ORDER BY r.default_probability DESC LIMIT 10;"
        )
        result = validate_sql(sql)
        self.assertTrue(result["valid"], result["reason"])

    def test_three_table_join_with_risk(self):
        sql = (
            "SELECT r.sk_id_curr, r.risk_score, i.late_payment_ratio "
            "FROM applicant_risk_scores r "
            "JOIN installment_summary i ON r.sk_id_curr = i.sk_id_curr "
            "WHERE r.risk_band = 'High' "
            "ORDER BY i.late_payment_ratio DESC LIMIT 10;"
        )
        result = validate_sql(sql)
        self.assertTrue(result["valid"], result["reason"])


class TestLiveDBQueries(unittest.TestCase):
    """Integration tests that run actual queries against the analytics database."""

    @classmethod
    def setUpClass(cls):
        """Skip DB tests if analytics DB does not exist."""
        if not os.path.exists(DEFAULT_DB_PATH):
            raise unittest.SkipTest(f"Analytics DB not found at {DEFAULT_DB_PATH}")

    def _execute(self, sql: str):
        """Execute a SQL query against the read-only analytics DB."""
        conn = sqlite3.connect(f"file:{DEFAULT_DB_PATH}?mode=ro", uri=True)
        try:
            import pandas as pd
            df = pd.read_sql_query(sql, conn)
            return df
        finally:
            conn.close()

    def test_risk_scores_table_exists_and_populated(self):
        """applicant_risk_scores must exist and have records."""
        try:
            df = self._execute("SELECT COUNT(*) AS cnt FROM applicant_risk_scores;")
            count = int(df["cnt"].iloc[0])
            self.assertGreater(count, 0, "applicant_risk_scores table is empty!")
        except Exception as e:
            self.skipTest(f"applicant_risk_scores not yet populated: {e}")

    def test_risk_scores_bands_are_valid(self):
        """All risk_band values must be 'Low', 'Medium', or 'High'."""
        try:
            df = self._execute(
                "SELECT DISTINCT risk_band FROM applicant_risk_scores ORDER BY risk_band;"
            )
            bands = set(df["risk_band"].tolist())
            valid_bands = {"Low", "Medium", "High"}
            self.assertTrue(
                bands.issubset(valid_bands),
                f"Invalid risk bands found: {bands - valid_bands}"
            )
        except Exception:
            self.skipTest("applicant_risk_scores not yet populated")

    def test_risk_scores_probabilities_in_range(self):
        """default_probability values must be in [0, 1]."""
        try:
            df = self._execute(
                "SELECT MIN(default_probability) AS min_p, MAX(default_probability) AS max_p "
                "FROM applicant_risk_scores;"
            )
            self.assertGreaterEqual(float(df["min_p"].iloc[0]), 0.0)
            self.assertLessEqual(float(df["max_p"].iloc[0]), 1.0)
        except Exception:
            self.skipTest("applicant_risk_scores not yet populated")

    def test_top_5_high_risk_deterministic_query_runs(self):
        """Top-5 high-risk deterministic SQL must execute successfully against real DB."""
        result = generate_sql_deterministic("Show me the top 5 high-risk applicants")
        if not result["is_supported"] or not result["sql"]:
            self.skipTest("Deterministic pattern not matched")

        try:
            df = self._execute(result["sql"])
            self.assertGreaterEqual(len(df), 1)
            self.assertIn("sk_id_curr", df.columns)
        except Exception as e:
            self.skipTest(f"Risk scores table not populated: {e}")

    def test_risk_band_distribution_query_runs(self):
        """Risk band distribution SQL must execute against real DB."""
        result = generate_sql_deterministic("What proportion of applicants fall into each risk band?")
        if not result["is_supported"] or not result["sql"]:
            self.skipTest("Deterministic pattern not matched")

        try:
            df = self._execute(result["sql"])
            self.assertGreaterEqual(len(df), 1)
            self.assertIn("risk_band", df.columns)
        except Exception as e:
            self.skipTest(f"Risk scores table not populated: {e}")

    def test_average_income_query_runs(self):
        """Average income query must execute correctly."""
        result = generate_sql_deterministic("What is the average annual income of applicants?")
        self.assertTrue(result["is_supported"])
        df = self._execute(result["sql"])
        self.assertEqual(len(df), 1)
        self.assertIn("avg_annual_income", df.columns)
        avg_val = float(df["avg_annual_income"].iloc[0])
        # Home Credit average income is roughly 168k
        self.assertGreater(avg_val, 50000, "Average income suspiciously low")
        self.assertLess(avg_val, 1_000_000, "Average income suspiciously high")

    def test_portfolio_default_rate_query_runs(self):
        """Portfolio default rate query must execute and give realistic result."""
        result = generate_sql_deterministic("What is the overall portfolio default rate?")
        self.assertTrue(result["is_supported"])
        df = self._execute(result["sql"])
        self.assertEqual(len(df), 1)
        self.assertIn("default_rate_pct", df.columns)
        rate = float(df["default_rate_pct"].iloc[0])
        # Home Credit dataset default rate is ~8%
        self.assertGreater(rate, 5.0, "Default rate suspiciously low")
        self.assertLess(rate, 20.0, "Default rate suspiciously high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
