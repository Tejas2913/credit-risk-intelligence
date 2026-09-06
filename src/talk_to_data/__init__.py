"""
Talk-to-Data Conversational Analytics Module
============================================
Provides Natural Language -> SQL generation, SQL validation guardrails,
and secure execution against the Credit Risk Analytics database.
"""

from src.talk_to_data.database import (
    DEFAULT_DB_PATH,
    DEFAULT_SCHEMA_PATH,
    execute_query,
    execute_query_safe,
    get_database_summary,
    init_database,
    is_read_only_query,
    populate_from_csv,
    populate_representative_sample
)
from src.talk_to_data.build_database import build_analytics_database
from src.talk_to_data.schema import (
    TABLE_NAME,
    SCHEMA_METADATA,
    get_allowed_columns,
    get_schema_prompt_text
)
from src.talk_to_data.prompts import (
    SQL_SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
    build_system_prompt,
    format_user_prompt
)
from src.talk_to_data.sql_validator import validate_sql
from src.talk_to_data.sql_generator import generate_sql
from src.talk_to_data.query_engine import QueryEngine, process_natural_language_query
from src.talk_to_data.llm import LLMClient
from src.talk_to_data.response_generator import generate_business_response
from src.talk_to_data.chatbot import answer_question, CreditRiskChatbot

__all__ = [
    "DEFAULT_DB_PATH",
    "DEFAULT_SCHEMA_PATH",
    "TABLE_NAME",
    "SCHEMA_METADATA",
    "SQL_SYSTEM_PROMPT",
    "FEW_SHOT_EXAMPLES",
    "get_allowed_columns",
    "get_schema_prompt_text",
    "build_system_prompt",
    "format_user_prompt",
    "init_database",
    "build_analytics_database",
    "populate_representative_sample",
    "populate_from_csv",
    "execute_query",
    "execute_query_safe",
    "get_database_summary",
    "is_read_only_query",
    "validate_sql",
    "generate_sql",
    "QueryEngine",
    "process_natural_language_query",
    "LLMClient",
    "generate_business_response",
    "answer_question",
    "CreditRiskChatbot"
]
