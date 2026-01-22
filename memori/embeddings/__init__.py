"""
Embeddings utilities.

The public entrypoints are:
- embed_texts
- embed_texts_async
- format_embedding_for_db
"""

from memori.embeddings._api import _embed_texts_async as embed_texts_async
from memori.embeddings._api import embed_texts
from memori.embeddings._format import format_embedding_for_db
from memori.embeddings._tei import TEI

__all__ = ["TEI", "embed_texts", "embed_texts_async", "format_embedding_for_db"]
