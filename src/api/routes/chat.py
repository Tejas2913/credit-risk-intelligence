"""
Module 3: Talk-to-Data Chat Route Handlers
===========================================
Exposes natural language to SQL portfolio analytics via `CreditRiskChatbot`.
Maintains conversation session context and executes read-only queries on `credit_risk_analytics.db`.
"""

import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_chatbot
from src.api.schemas import ChatRequest, ChatResponse
from src.talk_to_data.chatbot import CreditRiskChatbot

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Talk-to-Data NL-to-SQL"])


@router.post("", response_model=ChatResponse, summary="Ask an analytical question about the loan portfolio")
def chat_endpoint(
    request: ChatRequest
) -> ChatResponse:
    """
    Process a natural language business question:
    1. Identifies analytical intent and checks domain safety (redirects individual underwriting queries).
    2. Generates schema-grounded SQL query (LLM with deterministic fallback).
    3. Validates SQL strictly (SELECT-only, whitelisted table, no DDL/DML/injection).
    4. Executes query against read-only analytics database.
    5. Synthesizes business-readable executive answer.
    """
    question = request.message.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question message cannot be empty."
        )

    chatbot = get_chatbot(session_id=request.session_id)

    try:
        result = chatbot.ask(question)

        return ChatResponse(
            success=result.get("success", False),
            answer=result.get("answer", "Unable to generate a response."),
            sql=result.get("sql"),
            data=result.get("data", []),
            columns=result.get("columns", []),
            row_count=result.get("row_count", 0),
            intent=result.get("intent", "portfolio_analytics"),
            used_llm=result.get("used_llm", False),
            error=result.get("error")
        )
    except Exception as exc:
        logger.error("Chat processing exception for question '%s': %s", question, exc, exc_info=True)
        return ChatResponse(
            success=False,
            answer="An unexpected error occurred while processing your analytical request.",
            sql=None,
            data=[],
            columns=[],
            row_count=0,
            intent="unknown",
            used_llm=False,
            error="An internal error occurred while processing the chat request."
        )
