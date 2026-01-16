"""
Embeddings utilities.

The public entrypoints are:
- embed_texts
- format_embedding_for_db
"""

from memori.embeddings._api import embed_texts
from memori.embeddings._format import format_embedding_for_db

__all__ = ["embed_texts", "format_embedding_for_db"]
