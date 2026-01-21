from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True, slots=True)
class TEI:
    url: str
    timeout: int | None = 30
    headers: dict[str, str] | None = None

    def _request_headers(self) -> dict[str, str]:
        base = {"Content-Type": "application/json"}
        if self.headers:
            base.update(self.headers)
        return base

    def _post_embeddings(self, inputs: list[str], *, model: str) -> list[list[float]]:
        r = requests.post(
            self.url,
            headers=self._request_headers(),
            json={"input": inputs, "model": model},
            timeout=self.timeout,
        )
        r.raise_for_status()
        payload: Any = r.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise ValueError("TEI response missing 'data' list")

        out: list[list[float]] = []
        for item in data:
            if not isinstance(item, dict) or "embedding" not in item:
                raise ValueError("TEI response items must contain 'embedding'")
            emb = item["embedding"]
            if not isinstance(emb, list):
                raise ValueError("TEI embedding must be a list")
            out.append(emb)
        return out

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        return self._post_embeddings(texts, model=model)
