"""
Search utilities for Memori.

Public entrypoints:
- parse_embedding
- find_similar_embeddings
- search_entity_facts
"""

from memori.search._api import search_entity_facts
from memori.search._faiss import find_similar_embeddings
from memori.search._parsing import parse_embedding

__all__ = ["find_similar_embeddings", "parse_embedding", "search_entity_facts"]
