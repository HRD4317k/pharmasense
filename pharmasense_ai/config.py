"""Settings read from the environment / .env file (never hard-coded keys)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]

# OpenAI-compatible chat endpoints an ordinary person can use. All are optional: with LLM_PROVIDER=none
# (the default) the whole pipeline still runs, using deterministic rules only.
LLM_PRESETS = {
    "groq":       {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY",
                   "model": "llama-3.3-70b-versatile"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "key_env": "GEMINI_API_KEY",
                   "model": "gemini-2.5-flash"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY", "model": ""},
    "ollama":     {"base_url": "http://localhost:11434/v1", "key_env": "", "model": "llama3.1"},
}


def _clean(v: str | None) -> str:
    v = (v or "").strip().strip('"').strip("'")
    return "" if v.lower().startswith("your_") else v


@dataclass
class Settings:
    openfda_api_key: str = ""
    contact_email: str = ""
    llm_provider: str = "none"
    llm_api_key: str = ""
    llm_model: str = ""
    llm_base_url: str = ""
    cache_dir: Path = ROOT / ".cache"
    cache_ttl_days: float = 14.0
    timeout: float = 45.0

    @classmethod
    def load(cls, env_file: Path | None = None) -> "Settings":
        load_dotenv(env_file or ROOT / ".env")
        provider = _clean(os.environ.get("LLM_PROVIDER")).lower() or "none"
        preset = LLM_PRESETS.get(provider, {})
        key = _clean(os.environ.get("LLM_API_KEY")) or (_clean(os.environ.get(preset.get("key_env", ""))) if preset.get("key_env") else "")
        return cls(
            openfda_api_key=_clean(os.environ.get("OPENFDA_API_KEY")),
            contact_email=_clean(os.environ.get("CONTACT_EMAIL")),
            llm_provider=provider,
            llm_api_key=key,
            llm_model=_clean(os.environ.get("LLM_MODEL")) or preset.get("model", ""),
            llm_base_url=_clean(os.environ.get("LLM_BASE_URL")) or preset.get("base_url", ""),
            cache_dir=Path(_clean(os.environ.get("CACHE_DIR")) or ROOT / ".cache"),
            cache_ttl_days=float(_clean(os.environ.get("CACHE_TTL_DAYS")) or 14),
        )

    @property
    def llm_ready(self) -> bool:
        if self.llm_provider in ("", "none") or not self.llm_base_url or not self.llm_model:
            return False
        return bool(self.llm_api_key) or self.llm_provider == "ollama"

    def describe(self) -> dict:
        """What is configured (never prints key values)."""
        return {
            "openFDA key": "set" if self.openfda_api_key else "not set (anonymous: 1,000 requests/day)",
            "contact email": "set" if self.contact_email else "not set (recommended: sent in the User-Agent)",
            "LLM": (f"{self.llm_provider} / {self.llm_model}" if self.llm_ready else
                    "off - deterministic rules only (set LLM_PROVIDER in .env to enable)"),
            "cache": str(self.cache_dir),
        }
