from __future__ import annotations

from dataclasses import dataclass

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
        try:
            payload = r.json()
            data = payload["data"]
            if not isinstance(data, list):
                raise TypeError
            return [item["embedding"] for item in data]
        except Exception as e:
            raise ValueError("Invalid TEI response payload") from e

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        return self._post_embeddings(texts, model=model)
