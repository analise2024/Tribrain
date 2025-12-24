from __future__ import annotations

from dataclasses import dataclass

import requests


@dataclass
class OllamaClient:
    """Ollama /api/chat client.

    Requires a local Ollama server, e.g. `ollama serve`.
    """

    model: str = "llama3.1:8b-instruct"
    base_url: str = "http://localhost:11434"

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> str:
        url = self.base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
        }
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        data = r.json()
        # Ollama returns: {message: {role, content}, ...}
        msg = data.get("message") or {}
        return str(msg.get("content", ""))
