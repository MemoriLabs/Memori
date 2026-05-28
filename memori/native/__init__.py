"""Native Rust core helpers."""

from memori.native._adapter import RustCoreAdapter
from memori.native._embeddings import embed_texts
from memori.native._errors import RustCoreAdapterError

__all__ = [
    "RustCoreAdapter",
    "RustCoreAdapterError",
    "embed_texts",
]
