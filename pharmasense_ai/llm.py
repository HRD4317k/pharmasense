"""Optional LLM access through any OpenAI-compatible chat endpoint (Groq, Gemini, OpenRouter, Ollama).

The pipeline never *needs* the model: every LLM answer is parsed, then CHECKED - a hypothesis or critique that
cites an evidence id that does not exist is dropped and counted. The model can add reasoning but cannot add facts.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import requests

from .config import Settings


def extract_json(text: str | None):
    """First JSON object/array found in a model reply (models often wrap it in prose or ``` fences)."""
    if not text:
        return None
    t = re.sub(r"```(?:json)?", "", text)
    # try whichever bracket appears FIRST: a reply that is a list of objects must come back as the list, not its first object
    order = sorted((t.find(o), o, c) for o, c in (("{", "}"), ("[", "]")) if t.find(o) != -1)
    for i, opener, closer in order:
        j = t.rfind(closer)
        if j > i:
            try:
                return json.loads(t[i:j + 1])
            except ValueError:
                continue
    return None


class LLM:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.s = settings
        self.session = session or requests.Session()
        self.last_error = ""
        self.calls = 0
        self.dir = Path(settings.cache_dir) / "llm"

    @property
    def ready(self) -> bool:
        return self.s.llm_ready

    def chat(self, system: str, user: str, max_tokens: int = 1400, temperature: float = 0.1) -> str | None:
        if not self.ready:
            return None
        key = hashlib.sha1(json.dumps([self.s.llm_model, system, user, max_tokens, temperature]).encode()).hexdigest()
        path = self.dir / f"{key}.json"
        try:
            if path.exists():
                return json.loads(path.read_text())["text"]
        except Exception:
            pass
        headers = {"Content-Type": "application/json"}
        if self.s.llm_api_key:
            headers["Authorization"] = f"Bearer {self.s.llm_api_key}"
        body = {"model": self.s.llm_model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        url = self.s.llm_base_url.rstrip("/") + "/chat/completions"
        for attempt in range(3):
            try:
                self.calls += 1
                r = self.session.post(url, headers=headers, json=body, timeout=90)
            except requests.RequestException as e:
                self.last_error = f"network error: {e.__class__.__name__}"
                time.sleep(2 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                self.last_error = f"HTTP {r.status_code}"
                time.sleep(4 * (attempt + 1))
                continue
            if r.status_code != 200:
                self.last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                return None
            try:
                text = r.json()["choices"][0]["message"]["content"]
            except (KeyError, IndexError, ValueError):
                self.last_error = "unexpected response shape"
                return None
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({"text": text}))
            except Exception:
                pass
            return text
        return None
