from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LoCoMoTurn:
    speaker: str
    text: str
    turn_id: str | None = None
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class LoCoMoSession:
    session_id: str | None
    turns: tuple[LoCoMoTurn, ...]
    date_time: str | None = None


@dataclass(frozen=True, slots=True)
class LoCoMoQA:
    question_id: str
    question: str
    answers: tuple[str, ...]
    category: str | None = None
    evidence: Any | None = None


@dataclass(frozen=True, slots=True)
class LoCoMoSample:
    sample_id: str
    sessions: tuple[LoCoMoSession, ...]
    qa: tuple[LoCoMoQA, ...]
