from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

import requests

logger = logging.getLogger(__name__)


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
        t0 = perf_counter()
        r = requests.post(
            self.url,
            headers=self._request_headers(),
            json={"input": inputs, "model": model},
            timeout=self.timeout,
        )
        t_post = perf_counter()
        r.raise_for_status()
        t_raise = perf_counter()
        try:
            payload = r.json()
            t_json = perf_counter()
            data = payload["data"]
            if not isinstance(data, list):
                raise TypeError
            embeddings = [item["embedding"] for item in data]
            t_extract = perf_counter()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "TEI._post_embeddings: inputs=%d post=%.2fms raise=%.2fms json=%.2fms extract=%.2fms total=%.2fms",
                    len(inputs),
                    (t_post - t0) * 1000.0,
                    (t_raise - t_post) * 1000.0,
                    (t_json - t_raise) * 1000.0,
                    (t_extract - t_json) * 1000.0,
                    (t_extract - t0) * 1000.0,
                )
            return embeddings
        except Exception as e:
            raise ValueError("Invalid TEI response payload") from e

    def embed(self, texts: list[str], *, model: str) -> list[list[float]]:
        if not texts:
            return []
        return self._post_embeddings(texts, model=model)
