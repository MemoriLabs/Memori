from __future__ import annotations


def hit_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant or not retrieved or k <= 0:
        return 0.0
    top = retrieved[:k]
    return 1.0 if any(item in relevant for item in top) else 0.0


def mrr(relevant: set[str], retrieved: list[str]) -> float:
    if not relevant or not retrieved:
        return 0.0
    for i, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / float(i)
    return 0.0
