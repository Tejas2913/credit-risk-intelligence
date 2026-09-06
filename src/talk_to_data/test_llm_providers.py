"""
LLM Provider Integration & Safety Test Suite
============================================
Tests offline fallback, timeout handling, SQL validation, and optional live API
calls for Groq and OpenRouter when valid credentials are present in the environment.
"""

import os
import unittest
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.talk_to_data.chatbot import CreditRiskChatbot, answer_question
from src.talk_to_data.llm import LLMClient
from src.talk_to_data.sql_generator import generate_sql
from src.talk_to_data.sql_validator import validate_sql


class TestLLMProviders(unittest.TestCase):
    """Test suite covering LLM client abstraction, security, and live provider execution."""

    # ------------------------------------------------------------------
    # 1. Offline & Fallback Tests (Zero API Key Dependency)
    # ------------------------------------------------------------------

    def test_01_deterministic_offline_fallback(self):
        """Verify 100% offline functionality when provider is none or no keys are configured."""
        client = LLMClient(provider="none")
        self.assertFalse(client.is_available())
        info = client.get_provider_info()
        self.assertEqual(info["provider"], "none")
        self.assertTrue(info["fallback"])

        # Chatbot should answer cleanly via deterministic rules
        res = answer_question("What is the default rate?", llm_client=client, use_llm=False)
        self.assertTrue(res["success"])
        self.assertFalse(res["used_llm"])
        self.assertIn("8.07%", res["answer"])
        self.assertIsNotNone(res["sql"])
        self.assertGreater(len(res["data"]), 0)

    def test_02_simulated_network_failure_fallback(self):
        """Verify graceful fallback when LLM HTTP request raises Timeout or URLError."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.is_available.return_value = True
        mock_client.complete.side_effect = TimeoutError("Simulated LLM Gateway Timeout")

        res = answer_question("What is the default rate by education level?", llm_client=mock_client, use_llm=True)
        self.assertTrue(res["success"])
        self.assertFalse(res["used_llm"])
        self.assertIn("Lower secondary", res["answer"])
        self.assertIn("NAME_EDUCATION_TYPE", res["sql"].upper())

    def test_03_simulated_malformed_llm_json(self):
        """Verify graceful recovery when LLM returns invalid JSON or non-structured text."""
        mock_client = MagicMock(spec=LLMClient)
        mock_client.is_available.return_value = True
        mock_client.complete.return_value = "Here is your SQL query: SELECT * FROM table (MALFORMED JSON)"

        res = answer_question("What is the default rate by education level?", llm_client=mock_client, use_llm=True)
        self.assertTrue(res["success"])
        self.assertFalse(res["used_llm"])
        self.assertIsNotNone(res["sql"])

    def test_04_simulated_rogue_llm_malicious_sql_interception(self):
        """Verify that any SQL returned by an LLM is strictly checked by sql_validator.py."""
        malicious_sql = "SELECT * FROM applicant_analytics; DROP TABLE applicant_analytics;"
        validation = validate_sql(malicious_sql)
        self.assertFalse(validation["valid"])
        self.assertIn("semicolon", validation["reason"].lower())

    def test_05_underwriting_redirect_guardrail(self):
        """Verify individual loan decision questions are intercepted before calling any LLM."""
        res = answer_question("Should we approve loan application 396899?", use_llm=True)
        self.assertFalse(res["success"])
        self.assertIn("redirect", res["intent"])
        self.assertIn("Credit Risk", res["answer"])
        self.assertIsNone(res["sql"])

    def test_06_unsupported_domain_guardrail(self):
        """Verify out-of-domain macro questions are blocked cleanly."""
        res = answer_question("What was the national GDP growth rate?", use_llm=True)
        self.assertFalse(res["success"])
        self.assertEqual(res["intent"], "unsupported_domain")
        self.assertIn("GDP", res["answer"].upper())

    def test_07_provider_info_never_exposes_secrets(self):
        """Verify diagnostic provider info dict contains no API keys or secrets."""
        client = LLMClient()
        info = client.get_provider_info()
        self.assertIn("provider", info)
        self.assertIn("model", info)
        self.assertIn("is_available", info)
        self.assertNotIn("api_key", str(info).lower())
        self.assertNotIn("authorization", str(info).lower())

    # ------------------------------------------------------------------
    # 2. Live Provider Tests (Executes when valid API keys exist)
    # ------------------------------------------------------------------

    def test_08_live_groq_query_execution(self):
        """Test end-to-end NL -> SQL -> Execution -> Answer pipeline using live Groq API."""
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if not groq_key or groq_key.startswith("your_"):
            self.skipTest("GROQ_API_KEY not configured. Skipping live Groq test.")

        client = LLMClient(provider="groq")
        self.assertTrue(client.is_available(), "Groq client reported unavailable despite key presence.")

        cb = CreditRiskChatbot(use_llm=True)
        cb.llm_client = client

        res = cb.ask("What is the default rate across education levels?")
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["sql"])
        self.assertGreater(len(res["data"]), 0)
        self.assertIn("default", res["answer"].lower())

    def test_09_live_openrouter_query_execution(self):
        """Test end-to-end NL -> SQL -> Execution -> Answer pipeline using live OpenRouter API."""
        or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not or_key or or_key.startswith("your_"):
            self.skipTest("OPENROUTER_API_KEY not configured. Skipping live OpenRouter test.")

        client = LLMClient(provider="openrouter")
        self.assertTrue(client.is_available(), "OpenRouter client reported unavailable despite key presence.")

        cb = CreditRiskChatbot(use_llm=True)
        cb.llm_client = client

        res = cb.ask("What is the default rate across education levels?")
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["sql"])
        self.assertGreater(len(res["data"]), 0)
        self.assertIn("default", res["answer"].lower())


if __name__ == "__main__":
    unittest.main(verbosity=2)
