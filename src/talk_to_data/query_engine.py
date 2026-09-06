"""
Talk-to-Data Query Engine Module
================================
Orchestrates the complete Natural Language -> SQL Generation -> SQL Validation ->
Safe SQLite Execution pipeline.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.talk_to_data.database import DEFAULT_DB_PATH, execute_query
from src.talk_to_data.sql_generator import generate_sql
from src.talk_to_data.sql_validator import validate_sql


class QueryEngine:
    """
    End-to-end engine for processing conversational natural language questions
    against the Credit Risk Analytics database.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize QueryEngine with target database path."""
        self.db_path = db_path or DEFAULT_DB_PATH

    def process_query(
        self,
        question: str,
        conversation_context: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Execute the full pipeline:
        1. NL -> SQL Generation
        2. Security and Schema Validation
        3. Read-Only Execution against SQLite
        4. Structured Result Packaging

        Args:
            question: User natural-language question.
            conversation_context: Optional previous conversation turns.

        Returns:
            Dictionary with success status, sql, results, column headers, and metadata.
        """
        if not question or not question.strip():
            return {
                "success": False,
                "question": question,
                "sql": None,
                "intent": "empty_query",
                "explanation": None,
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": "Query cannot be empty."
            }

        # Step 1: Generate SQL from natural language
        gen_result = generate_sql(question, conversation_context)

        if not gen_result.get("is_supported", False) or not gen_result.get("sql"):
            return {
                "success": False,
                "question": question,
                "sql": None,
                "intent": gen_result.get("intent", "unsupported"),
                "explanation": gen_result.get("explanation", "Question cannot be answered from the schema."),
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": gen_result.get("explanation", "Unsupported question.")
            }

        raw_sql = gen_result["sql"]

        # Step 2: Validate SQL syntax, tokens, and safety guardrails
        val_result = validate_sql(raw_sql, db_path=self.db_path)

        if not val_result.get("valid", False):
            return {
                "success": False,
                "question": question,
                "sql": raw_sql,
                "intent": gen_result.get("intent", "validation_failure"),
                "explanation": gen_result.get("explanation"),
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": f"SQL validation failed: {val_result.get('reason')}"
            }

        sanitized_sql = val_result["sanitized_sql"]

        # Step 3: Execute validated read-only SQL against SQLite
        try:
            df = execute_query(sanitized_sql, db_path=self.db_path)
        except Exception as e:
            return {
                "success": False,
                "question": question,
                "sql": sanitized_sql,
                "intent": gen_result.get("intent", "execution_error"),
                "explanation": gen_result.get("explanation"),
                "columns": [],
                "rows": [],
                "row_count": 0,
                "error": "A database error occurred while executing the query."
            }

        # Step 4: Package clean response
        return {
            "success": True,
            "question": question,
            "sql": sanitized_sql,
            "intent": gen_result.get("intent"),
            "explanation": gen_result.get("explanation"),
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
            "row_count": len(df),
            "error": None
        }


def process_natural_language_query(
    question: str,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    db_path: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience functional wrapper for QueryEngine."""
    engine = QueryEngine(db_path=db_path)
    return engine.process_query(question, conversation_context)
