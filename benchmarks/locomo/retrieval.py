from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from benchmarks.locomo._types import LoCoMoSample


@dataclass(frozen=True, slots=True)
class TurnFact:
    turn_id: str
    content: str


_TURN_ID_RE = re.compile(r"^\[(?P<turn_id>[^\]]+)\]\s*")


def build_turn_facts(
    sample: LoCoMoSample,
) -> tuple[tuple[TurnFact, ...], dict[tuple[str, int], str]]:
    facts: list[TurnFact] = []
    index: dict[tuple[str, int], str] = {}

    for s_idx, session in enumerate(sample.sessions):
        session_id = session.session_id or f"session-{s_idx}"
        for t_idx, turn in enumerate(session.turns):
            turn_id = turn.turn_id or f"{sample.sample_id}:{session_id}:{t_idx}"
            index[(session_id, t_idx)] = turn_id

            speaker = (turn.speaker or "").strip()
            prefix = f"[{turn_id}]"
            body = f"{speaker}: {turn.text}" if speaker else turn.text
            if session.date_time:
                body = f"{body} (session_time: {session.date_time})"
            if turn.timestamp:
                body = f"{body} (ts: {turn.timestamp})"
            facts.append(TurnFact(turn_id=turn_id, content=f"{prefix} {body}".strip()))

    return tuple(facts), index


def extract_turn_id_from_content(content: str) -> str | None:
    m = _TURN_ID_RE.match(content or "")
    if not m:
        return None
    v = m.group("turn_id").strip()
    return v or None


def evidence_to_turn_ids(
    evidence: Any, *, turn_index: dict[tuple[str, int], str]
) -> set[str]:
    if evidence is None:
        return set()

    out: set[str] = set()

    if isinstance(evidence, dict):
        _add_evidence_obj(evidence, out=out, turn_index=turn_index)
        return out

    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict):
                _add_evidence_obj(item, out=out, turn_index=turn_index)
            elif isinstance(item, int):
                # If evidence is a bare turn index, only safe when there's a single session.
                _add_evidence_turn_index(item, out=out, turn_index=turn_index)
            elif isinstance(item, str):
                if item:
                    out.add(item)
        return out

    if isinstance(evidence, int):
        _add_evidence_turn_index(evidence, out=out, turn_index=turn_index)
        return out

    if isinstance(evidence, str) and evidence:
        out.add(evidence)

    return out


def _add_evidence_obj(
    obj: dict[str, Any], *, out: set[str], turn_index: dict[tuple[str, int], str]
) -> None:
    session_id = _coerce_str(obj.get("session_id")) or _coerce_str(obj.get("session"))
    turn_idx = _coerce_int(obj.get("turn_index"))
    if session_id is not None and turn_idx is not None:
        tid = turn_index.get((session_id, turn_idx))
        if tid:
            out.add(tid)
        return

    turn_id = _coerce_str(obj.get("turn_id")) or _coerce_str(obj.get("evidence_id"))
    if turn_id:
        out.add(turn_id)


def _add_evidence_turn_index(
    idx: int, *, out: set[str], turn_index: dict[tuple[str, int], str]
) -> None:
    sessions = {session_id for (session_id, _), _tid in turn_index.items()}
    if len(sessions) != 1:
        return
    (session_id,) = sessions
    tid = turn_index.get((session_id, idx))
    if tid:
        out.add(tid)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value).strip() or None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except Exception:
        return None
