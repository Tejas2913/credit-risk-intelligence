"""
API Dependencies & Shared Service Singletons
=============================================
Manages cached, thread-safe instances of:
- CreditRiskPredictor
- CreditRiskExplainer
- CreditRiskChatbot session registry
"""

import logging
import os
import sqlite3
import threading
from collections import OrderedDict
from typing import Dict, Optional

from src.ml.explainer import CreditRiskExplainer
from src.ml.predictor import CreditRiskPredictor
from src.talk_to_data.chatbot import CreditRiskChatbot

logger = logging.getLogger(__name__)

# Global singletons
_predictor_instance: Optional[CreditRiskPredictor] = None
_explainer_instance: Optional[CreditRiskExplainer] = None
_chatbots: OrderedDict = OrderedDict()          # LRU-ordered session registry
_chatbots_lock = threading.Lock()               # Guards concurrent access
MAX_SESSIONS = 1000                             # Hard cap on in-memory sessions
DEFAULT_SESSION_ID = "default"


def get_predictor() -> CreditRiskPredictor:
    """Retrieve or initialize singleton CreditRiskPredictor instance."""
    global _predictor_instance
    if _predictor_instance is None:
        logger.info("Initializing singleton CreditRiskPredictor...")
        _predictor_instance = CreditRiskPredictor()
    return _predictor_instance


def get_explainer() -> CreditRiskExplainer:
    """Retrieve or initialize singleton CreditRiskExplainer instance."""
    global _explainer_instance
    if _explainer_instance is None:
        predictor = get_predictor()
        logger.info("Initializing singleton CreditRiskExplainer...")
        _explainer_instance = CreditRiskExplainer(predictor=predictor)
    return _explainer_instance


def get_chatbot(session_id: Optional[str] = None) -> CreditRiskChatbot:
    """
    Retrieve or create a CreditRiskChatbot instance for the specified session ID.
    Supports session isolation and multi-turn context retention.
    Bounded to MAX_SESSIONS (1000) using FIFO eviction of the oldest session.
    """
    sid = session_id.strip() if session_id and session_id.strip() else DEFAULT_SESSION_ID
    with _chatbots_lock:
        if sid in _chatbots:
            # Move to end to mark as most-recently-used
            _chatbots.move_to_end(sid)
            return _chatbots[sid]

        # Evict oldest session(s) when limit is reached
        while len(_chatbots) >= MAX_SESSIONS:
            oldest_sid, _ = next(iter(_chatbots.items()))
            del _chatbots[oldest_sid]
            logger.info("Session registry full; evicted oldest session: %s", oldest_sid)

        _chatbots[sid] = CreditRiskChatbot(max_history_turns=5, use_llm=True)
        return _chatbots[sid]


def check_db_health(db_path: str, table_name: str) -> bool:
    """Check if SQLite database file exists and is accessible."""
    if not os.path.exists(db_path):
        return False
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cur = conn.cursor()
        cur.execute(f"SELECT 1 FROM {table_name} LIMIT 1;")
        cur.fetchone()
        conn.close()
        return True
    except Exception as exc:
        logger.error("Health check failed for %s: %s", db_path, exc)
        return False
