"""
LLM Provider Abstraction Module
===============================
Provides a unified, lightweight interface for multiple LLM providers:
- Groq (fast inference, e.g. llama-3.3-70b-versatile)
- OpenRouter (unified multi-model gateway, e.g. meta-llama/llama-3.3-70b-instruct)
- OpenAI (gpt-4o-mini, gpt-4o)
- Anthropic (claude-3-5-sonnet, claude-3-haiku)
- Google Gemini (gemini-1.5-flash, gemini-2.0-flash)
- Ollama (local llama3, mistral)

Implements graceful fallback: if no valid API key is present, network fails,
or timeouts occur, methods return None to enable deterministic fallback without crashing.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

# Automatically load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Placeholder patterns indicating an unconfigured key
PLACEHOLDER_KEYS = {
    "your_openai_api_key_here",
    "your_groq_api_key_here",
    "your_openrouter_api_key_here",
    "your_anthropic_api_key_here",
    "your_gemini_api_key_here",
    "none",
    ""
}


def _is_valid_key(key: Optional[str]) -> bool:
    """Check if an API key is non-empty and not a placeholder."""
    if not key:
        return False
    return key.strip().lower() not in PLACEHOLDER_KEYS


class LLMClient:
    """
    Unified LLM Client supporting Groq, OpenRouter, OpenAI, Anthropic, Gemini, and Ollama
    using standard HTTP requests to avoid hard dependency conflicts.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        timeout: float = 12.0
    ):
        self.provider = (provider or os.getenv("LLM_PROVIDER", "")).strip().lower()
        self.temperature = float(temperature or os.getenv("LLM_TEMPERATURE", 0.0))
        self.timeout = timeout

        # API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        self.ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

        # Auto-detect provider if not explicitly specified
        if not self.provider:
            self.provider = self._auto_detect_provider()
        elif self.provider != "none" and not self._has_key_for_provider(self.provider):
            self.provider = self._auto_detect_provider()

        # Model selection precedence: parameter -> provider-specific env -> generic LLM_MODEL -> default
        self.model = model or self._resolve_model_for_provider(self.provider)

    def _has_key_for_provider(self, provider: str) -> bool:
        """Check if active credentials exist for the chosen provider."""
        if provider == "none":
            return False
        elif provider == "groq":
            return _is_valid_key(self.groq_api_key)
        elif provider == "openrouter":
            return _is_valid_key(self.openrouter_api_key)
        elif provider == "openai":
            return _is_valid_key(self.openai_api_key)
        elif provider == "anthropic":
            return _is_valid_key(self.anthropic_api_key)
        elif provider == "gemini":
            return _is_valid_key(self.gemini_api_key)
        elif provider == "ollama":
            return True
        return False

    def _auto_detect_provider(self) -> str:
        """Auto-detect provider based on available environment variables."""
        if _is_valid_key(self.groq_api_key):
            return "groq"
        if _is_valid_key(self.openrouter_api_key):
            return "openrouter"
        if _is_valid_key(self.openai_api_key):
            return "openai"
        if _is_valid_key(self.gemini_api_key):
            return "gemini"
        if _is_valid_key(self.anthropic_api_key):
            return "anthropic"
        return "none"

    def _resolve_model_for_provider(self, provider: str) -> str:
        """Resolve model name using provider-specific env, generic LLM_MODEL, or defaults."""
        generic_model = os.getenv("LLM_MODEL", "").strip()

        if provider == "groq":
            return os.getenv("GROQ_MODEL", "").strip() or generic_model or "openai/gpt-oss-120b"
        elif provider == "openrouter":
            return os.getenv("OPENROUTER_MODEL", "").strip() or generic_model or "meta-llama/llama-3.3-70b-instruct"
        elif provider == "openai":
            return os.getenv("OPENAI_MODEL", "").strip() or generic_model or "gpt-4o-mini"
        elif provider == "anthropic":
            return os.getenv("ANTHROPIC_MODEL", "").strip() or generic_model or "claude-3-5-sonnet-20241022"
        elif provider == "gemini":
            return os.getenv("GEMINI_MODEL", "").strip() or generic_model or "gemini-1.5-flash"
        elif provider == "ollama":
            return os.getenv("OLLAMA_MODEL", "").strip() or generic_model or "llama3"

        return generic_model or ""

    def is_available(self) -> bool:
        """Return True if a valid provider configuration is active."""
        if self.provider == "none" or not self.provider:
            return False
        return self._has_key_for_provider(self.provider)

    def get_provider_info(self) -> Dict[str, Any]:
        """Return safe diagnostic metadata without exposing secrets."""
        return {
            "provider": self.provider,
            "model": self.model,
            "is_available": self.is_available(),
            "fallback": not self.is_available()
        }

    def complete(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        json_mode: bool = False
    ) -> Optional[str]:
        """
        Execute completion with selected provider.
        Returns the raw response string, or None if the request failed or no provider is configured.
        """
        if not self.is_available():
            return None

        try:
            if self.provider == "groq":
                return self._call_openai_compatible(
                    endpoint="https://api.groq.com/openai/v1/chat/completions",
                    api_key=self.groq_api_key,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode
                )
            elif self.provider == "openrouter":
                return self._call_openrouter(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode
                )
            elif self.provider == "openai":
                return self._call_openai_compatible(
                    endpoint="https://api.openai.com/v1/chat/completions",
                    api_key=self.openai_api_key,
                    prompt=prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode
                )
            elif self.provider == "anthropic":
                return self._call_anthropic(
                    prompt=prompt,
                    system_prompt=system_prompt
                )
            elif self.provider == "gemini":
                return self._call_gemini(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode
                )
            elif self.provider == "ollama":
                return self._call_ollama(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    json_mode=json_mode
                )
        except Exception as e:
            logger.warning("LLM call failed for provider %s: %s. Falling back to deterministic engine.", self.provider, e)
            return None

        return None

    def _call_openai_compatible(
        self,
        endpoint: str,
        api_key: str,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool,
        extra_headers: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "CreditRiskIntelligence/1.0"
        }
        if extra_headers:
            headers.update(extra_headers)

        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _call_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool
    ) -> Optional[str]:
        """Call OpenRouter API with custom headers and JSON formatting support."""
        extra_headers = {
            "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost:8000"),
            "X-Title": "Credit Risk Intelligence Platform"
        }
        return self._call_openai_compatible(
            endpoint="https://openrouter.ai/api/v1/chat/completions",
            api_key=self.openrouter_api_key,
            prompt=prompt,
            system_prompt=system_prompt,
            json_mode=json_mode,
            extra_headers=extra_headers
        )

    def _call_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str]
    ) -> Optional[str]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": 1024,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.anthropic_api_key,
            "anthropic-version": "2023-06-01"
        }

        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["content"][0]["text"]

    def _call_gemini(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool
    ) -> Optional[str]:
        endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.gemini_api_key}"
        
        if system_prompt:
            full_prompt = f"System Instructions:\n{system_prompt}\n\nUser Request:\n{prompt}"
        else:
            full_prompt = prompt

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature
            }
        }
        if json_mode:
            payload["generationConfig"]["responseMimeType"] = "application/json"

        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        json_mode: bool
    ) -> Optional[str]:
        endpoint = f"{self.ollama_base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "stream": False,
            "options": {"temperature": self.temperature}
        }
        if json_mode:
            payload["format"] = "json"

        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("response", "")
