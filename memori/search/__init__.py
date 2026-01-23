"""
Search utilities for Memori.

Public entrypoints:
- parse_embedding
- find_similar_embeddings
- search_facts
- FactCandidate
- FactCandidates
- FactSearchResult
"""

from memori.search._api import search_facts
from memori.search._faiss import find_similar_embeddings
from memori.search._parsing import parse_embedding
from memori.search._types import FactCandidate, FactCandidates, FactSearchResult

__all__ = [
    "find_similar_embeddings",
    "parse_embedding",
    "search_facts",
    "FactCandidate",
    "FactCandidates",
    "FactSearchResult",
]
