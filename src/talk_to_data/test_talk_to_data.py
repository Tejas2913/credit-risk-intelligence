"""
Talk-to-Data Validation and Test Suite
======================================
Tests the 7 core analytical patterns, malicious SQL rejection, hallucination controls,
and individual decision redirection.
"""

import os
import sys
import pandas as pd

from src.talk_to_data.query_engine import QueryEngine
from src.talk_to_data.sql_validator import validate_sql
from src.talk_to_data.database import DEFAULT_DB_PATH

def run_tests() -> None:
    print("=" * 80)
    print("TALK-TO-DATA COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print(f"Testing against Database: {DEFAULT_DB_PATH}")
    assert os.path.exists(DEFAULT_DB_PATH), f"Database not found at {DEFAULT_DB_PATH}"

    engine = QueryEngine(db_path=DEFAULT_DB_PATH)

    # =========================================================================
    # 1. Test All 7 Required Natural-Language Query Patterns
    # =========================================================================
    test_questions = [
        ("Pattern 1 - Overall Portfolio", "What is the overall default rate and total applicant count?"),
        ("Pattern 2 - Education Level", "What is the default rate by education level?"),
        ("Pattern 3 - Income Tiers", "How does default risk vary across income tiers?"),
        ("Pattern 4 - Refusal History", "Do applicants with previous loan refusals have higher default rates?"),
        ("Pattern 5 - Leverage Exposure", "What is the default rate for highly leveraged applicants?"),
        ("Pattern 6 - Late Payment Behavior", "How does late payment history relate to default risk?"),
        ("Pattern 7 - Organization Risk", "Which organization types have the highest default rates?")
    ]

    print("\n" + "=" * 80)
    print("1. TESTING 7 NATURAL-LANGUAGE QUERY PATTERNS")
    print("=" * 80)

    for label, q in test_questions:
        print(f"\n[{label}]")
        print(f"Question : \"{q}\"")
        result = engine.process_query(q)

        print(f"Success  : {result['success']}")
        print(f"Intent   : {result['intent']}")
        print(f"SQL      : {result['sql']}")
        print(f"Row Count: {result['row_count']}")
        assert result["success"] is True, f"Failed for query: {q} -> {result['error']}"
        assert result["row_count"] > 0, f"No rows returned for: {q}"
        
        # Display small tabular preview
        df = pd.DataFrame(result["rows"])
        print(df.to_string(index=False))

    # =========================================================================
    # 2. Test Malicious SQL Injection & Mutation Rejection
    # =========================================================================
    print("\n" + "=" * 80)
    print("2. TESTING MALICIOUS SQL REJECTION & SECURITY GUARDRAILS")
    print("=" * 80)

    malicious_inputs = [
        "DROP TABLE applicant_analytics;",
        "DELETE FROM applicant_analytics WHERE sk_id_curr = 100001;",
        "UPDATE applicant_analytics SET target = 0;",
        "INSERT INTO applicant_analytics (sk_id_curr, target, amt_income_total, amt_credit, name_contract_type) VALUES (999999, 1, 100000, 200000, 'Cash loans');",
        "PRAGMA table_info(applicant_analytics);",
        "SELECT * FROM applicant_analytics; DROP TABLE applicant_analytics;",
        "SELECT * FROM sqlite_master;",
        "-- comment\nDROP TABLE applicant_analytics;"
    ]

    for malicious_sql in malicious_inputs:
        val = validate_sql(malicious_sql, db_path=DEFAULT_DB_PATH)
        print(f"Test SQL : {malicious_sql.replace(chr(10), ' ')}")
        print(f"Valid    : {val['valid']}")
        print(f"Reason   : {val['reason']}")
        assert val["valid"] is False, f"Security Failure: Malicious SQL was NOT rejected -> {malicious_sql}"
        print("Status   : PASSED (Safely Blocked)\n")

    # =========================================================================
    # 3. Test Unsupported Questions & Hallucination Prevention
    # =========================================================================
    print("=" * 80)
    print("3. TESTING UNSUPPORTED QUESTIONS & HALLUCINATION CONTROL")
    print("=" * 80)

    unsupported_questions = [
        "What was the national unemployment rate last year?",
        "What is the applicant's credit score from Experian?",
        "What is the current inflation rate and GDP growth?"
    ]

    for uq in unsupported_questions:
        print(f"Question : \"{uq}\"")
        res = engine.process_query(uq)
        print(f"Success  : {res['success']}")
        print(f"Intent   : {res['intent']}")
        print(f"Message  : {res['error']}")
        assert res["success"] is False, f"Hallucination Failure: Unsupported question was executed -> {uq}"
        assert "cannot be answered" in res["error"], "Expected controlled refusal message."
        print("Status   : PASSED (Controlled Response without Hallucination)\n")

    # =========================================================================
    # 4. Test Individual Underwriting Decision Redirection
    # =========================================================================
    print("=" * 80)
    print("4. TESTING INDIVIDUAL UNDERWRITING SAFETY REDIRECTION")
    print("=" * 80)

    underwriting_questions = [
        "Should applicant 100001 be approved for a loan?",
        "Should I reject this applicant based on their profile?"
    ]

    for dq in underwriting_questions:
        print(f"Question : \"{dq}\"")
        res = engine.process_query(dq)
        print(f"Success  : {res['success']}")
        print(f"Intent   : {res['intent']}")
        print(f"Message  : {res['error']}")
        assert res["success"] is False
        assert "Credit Risk ML Predictor" in res["error"]
        print("Status   : PASSED (Safely Redirected to ML Predictor)\n")

    # =========================================================================
    # 5. Test Conversation Context (Follow-up Turn)
    # =========================================================================
    print("=" * 80)
    print("5. TESTING CONVERSATION CONTEXT HANDLING")
    print("=" * 80)

    # First turn: Education
    t1_q = "What is the default rate by education level?"
    t1_res = engine.process_query(t1_q)
    print(f"Turn 1 Question : \"{t1_q}\"")
    print(f"Turn 1 SQL      : {t1_res['sql']}")

    # Second turn: Add income filter via context
    t2_q = "What about only applicants earning above 200000?"
    context = [{"question": t1_q, "sql": t1_res["sql"]}]
    t2_res = engine.process_query(t2_q, conversation_context=context)
    print(f"\nTurn 2 Question : \"{t2_q}\"")
    print(f"Turn 2 SQL      : {t2_res['sql']}")
    print(f"Turn 2 Row Count: {t2_res['row_count']}")
    assert t2_res["success"] is True
    assert "WHERE amt_income_total > 200000" in t2_res["sql"]
    print("Status          : PASSED (Context Filter Successfully Applied)\n")

    print("=" * 80)
    print("ALL TEST SUITE CHECKS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
