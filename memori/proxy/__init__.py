from __future__ import annotations


def __getattr__(name: str):  # type: ignore[override]
    if name == "MemoriCrewAIAdapter":
        from .crewai import MemoriCrewAIAdapter

        return MemoriCrewAIAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["MemoriCrewAIAdapter"]
