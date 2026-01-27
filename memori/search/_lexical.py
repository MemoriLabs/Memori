from __future__ import annotations

import logging
import math
import os
import re
from collections import Counter

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "there",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _tokenize(text: str) -> list[str]:
    tokens = [t for t in _TOKEN_RE.findall((text or "").lower()) if t]
    return [t for t in tokens if t not in _STOPWORDS]


def lexical_scores_for_ids(*, query_text: str, ids: list, content_map: dict) -> dict:
    """
    Compute a BM25 score in [0, 1] for each doc over the candidate pool.
    """
    q_tokens = _tokenize(query_text)
    if not q_tokens:
        return dict.fromkeys(ids, 0.0)

    docs_tf: dict = {}
    doc_len: dict = {}
    for i in ids:
        content = content_map.get(i, "")
        toks = _tokenize(content)
        docs_tf[i] = Counter(toks)
        doc_len[i] = len(toks)

    n_docs = len(ids)
    avgdl = (sum(doc_len.values()) / float(n_docs)) if n_docs else 0.0

    q_terms = set(q_tokens)
    df: dict[str, int] = {}
    for t in q_terms:
        df[t] = sum(1 for i in ids if docs_tf.get(i, Counter()).get(t, 0) > 0)

    k1 = 1.2
    b = 0.75

    def idf(t: str) -> float:
        dft = float(df.get(t, 0))
        return math.log(1.0 + ((n_docs - dft + 0.5) / (dft + 0.5)))

    raw: dict = {}
    for i in ids:
        tf = docs_tf.get(i, Counter())
        dl = float(doc_len.get(i, 0))
        denom_norm = (1.0 - b) + (b * (dl / avgdl)) if avgdl > 0 else 1.0
        score = 0.0
        for t in q_terms:
            f = float(tf.get(t, 0))
            if f <= 0.0:
                continue
            score += idf(t) * ((f * (k1 + 1.0)) / (f + (k1 * denom_norm)))
        raw[i] = score

    max_score = max(raw.values()) if raw else 0.0
    if max_score <= 0.0:
        return dict.fromkeys(ids, 0.0)

    return {i: float(raw.get(i, 0.0) / max_score) for i in ids}


def dense_lexical_weights(*, query_text: str) -> tuple[float, float]:
    """
    Return (w_cos, w_lex) for ranking.

    We bias toward lexical matching for very short queries where exact terms
    are usually high-signal.
    """
    q_tokens = _tokenize(query_text)

    try:
        w_lex = float(os.environ.get("MEMORI_RECALL_LEX_WEIGHT", "0.15") or "0.15")
    except ValueError:
        w_lex = 0.15
    if len(q_tokens) <= 2:
        try:
            w_lex = float(
                os.environ.get("MEMORI_RECALL_LEX_WEIGHT_SHORT", "0.30") or "0.30"
            )
        except ValueError:
            w_lex = 0.30
    w_lex = max(0.05, min(0.40, w_lex))
    return (1.0 - w_lex, w_lex)
