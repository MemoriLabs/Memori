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


def hit_at_k_groups(relevant: set[str], retrieved: list[set[str]], k: int) -> float:
    """
    Variant of hit@k where each retrieved rank can map to multiple IDs.

    This is useful when a retrieved item (e.g., an AA fact) may plausibly originate
    from multiple LoCoMo turns (dia_id).
    """
    if not relevant or not retrieved or k <= 0:
        return 0.0
    for group in retrieved[:k]:
        if group and (group & relevant):
            return 1.0
    return 0.0


def mrr_groups(relevant: set[str], retrieved: list[set[str]]) -> float:
    if not relevant or not retrieved:
        return 0.0
    for i, group in enumerate(retrieved, start=1):
        if group and (group & relevant):
            return 1.0 / float(i)
    return 0.0
