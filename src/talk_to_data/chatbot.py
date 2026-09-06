"""
Talk-to-Data Chatbot & High-Level Answering Interface
=====================================================
Coordinates the end-to-end analytical workflow:
1. Natural Language Question
2. Intent Safety & Guardrails (Underwriting redirect & unsupported domains)
3. Schema-Grounded SQL Generation (LLM with deterministic fallback)
4. Multi-Layer SQL Validation (Read-only, injection prevention)
5. Read-Only SQLite Query Execution
6. Business-Readable Executive Answer Synthesis (LLM with deterministic fallback)
"""

import logging
from typing import Any, Dict, List, Optional

from src.talk_to_data.database import execute_query_safe
from src.talk_to_data.llm import LLMClient
from src.talk_to_data.response_generator import generate_business_response
from src.talk_to_data.sql_generator import generate_sql
from src.talk_to_data.sql_validator import validate_sql

logger = logging.getLogger(__name__)


def answer_question(
    question: str,
    conversation_context: Optional[List[Dict[str, Any]]] = None,
    llm_client: Optional[LLMClient] = None,
    use_llm: bool = True
) -> Dict[str, Any]:
    """
    High-level, production-ready interface for asking natural-language analytics questions.

    Args:
        question: User query string.
        conversation_context: Optional list of previous turn dictionaries.
        llm_client: Optional custom LLMClient instance.
        use_llm: Whether to attempt LLM generation/synthesis if available.

    Returns:
        Structured response dictionary containing status, SQL, data, columns, and business answer.
    """
    client = llm_client or LLMClient()

    # Step 1: Generate SQL & check intent safety
    gen_result = generate_sql(
        question=question,
        conversation_context=conversation_context,
        llm_client=client,
        use_llm=use_llm
    )

    used_llm = gen_result.get("used_llm", False)
    intent = gen_result.get("intent", "portfolio_analytics")

    # Step 2: Handle unsupported queries or safety redirects
    if not gen_result.get("is_supported", False):
        return {
            "success": False,
            "question": question,
            "intent": intent,
            "sql": None,
            "answer": gen_result.get("explanation", "This question cannot be answered from the available analytics data."),
            "data": [],
            "columns": [],
            "row_count": 0,
            "used_llm": used_llm,
            "error": None
        }

    sql = gen_result.get("sql")
    if not sql:
        return {
            "success": False,
            "question": question,
            "intent": intent,
            "sql": None,
            "answer": "Unable to generate a valid SQL query for this question.",
            "data": [],
            "columns": [],
            "row_count": 0,
            "used_llm": used_llm,
            "error": "Missing SQL"
        }

    # Step 3: Validate SQL strictly before execution
    validation = validate_sql(sql)
    if not validation["valid"]:
        logger.warning("Generated SQL failed safety validation: %s. Reason: %s", sql, validation["reason"])
        return {
            "success": False,
            "question": question,
            "intent": intent,
            "sql": sql,
            "answer": "The generated analytical query failed safety validation checks.",
            "data": [],
            "columns": [],
            "row_count": 0,
            "used_llm": used_llm,
            "error": validation["reason"]
        }

    # Step 4: Execute query against read-only analytics database
    exec_result = execute_query_safe(sql)
    if not exec_result["success"]:
        logger.error("Database query execution failed: %s", exec_result.get("error"))
        return {
            "success": False,
            "question": question,
            "intent": intent,
            "sql": sql,
            "answer": "A database error occurred while calculating analytics.",
            "data": [],
            "columns": [],
            "row_count": 0,
            "used_llm": used_llm,
            "error": exec_result.get("error")
        }

    rows = exec_result.get("rows", [])
    columns = exec_result.get("columns", [])

    # Step 5: Synthesize executive business answer
    business_answer = generate_business_response(
        question=question,
        sql=sql,
        data=rows,
        columns=columns,
        intent=intent,
        llm_client=client if (use_llm and used_llm) else None
    )

    return {
        "success": True,
        "question": question,
        "intent": intent,
        "sql": sql,
        "answer": business_answer,
        "data": rows,
        "columns": columns,
        "row_count": len(rows),
        "used_llm": used_llm,
        "error": None
    }


class CreditRiskChatbot:
    """
    Lightweight stateful conversation manager for the Talk-to-Data platform.
    Retains a compact sliding window of recent conversation turns.
    """

    def __init__(self, max_history_turns: int = 5, use_llm: bool = True):
        self.max_history_turns = max_history_turns
        self.use_llm = use_llm
        self.history: List[Dict[str, Any]] = []
        self.llm_client = LLMClient()

    def ask(self, question: str) -> Dict[str, Any]:
        """
        Process a question in the context of the active conversation session.
        """
        result = answer_question(
            question=question,
            conversation_context=self.history,
            llm_client=self.llm_client,
            use_llm=self.use_llm
        )

        # Update conversation history
        turn_record = {
            "question": question,
            "intent": result.get("intent"),
            "sql": result.get("sql"),
            "answer": result.get("answer"),
            "success": result.get("success")
        }
        self.history.append(turn_record)

        # Keep history compact
        if len(self.history) > self.max_history_turns:
            self.history = self.history[-self.max_history_turns:]

        return result

    def clear(self) -> None:
        """Clear conversation context history."""
        self.history.clear()

    def get_history(self) -> List[Dict[str, Any]]:
        """Retrieve recent conversation turns."""
        return list(self.history)
