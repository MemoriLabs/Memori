from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from benchmarks.locomo._types import LoCoMoQA, LoCoMoSample, LoCoMoSession, LoCoMoTurn


def load_locomo_json(path: str | Path) -> list[LoCoMoSample]:
    raw = Path(path).read_text(encoding="utf-8")
    data = json.loads(raw)

    if not isinstance(data, list):
        raise ValueError("LoCoMo JSON must be a list of samples")

    samples: list[LoCoMoSample] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Sample at index {i} must be an object")

        sample_id = _coerce_str(item.get("sample_id")) or _coerce_str(item.get("id"))
        if not sample_id:
            raise ValueError(f"Sample at index {i} is missing 'sample_id'")

        sessions = _parse_conversation(item.get("conversation"))
        qa = _parse_qa(item.get("qa"))
        samples.append(LoCoMoSample(sample_id=sample_id, sessions=sessions, qa=qa))

    return samples


def _parse_conversation(value: Any) -> tuple[LoCoMoSession, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):
        return _parse_conversation_dict(value)
    if not isinstance(value, list):
        raise ValueError("'conversation' must be a list of sessions or an object")

    out: list[LoCoMoSession] = []
    for idx, session in enumerate(value):
        if not isinstance(session, dict):
            raise ValueError(f"conversation[{idx}] must be an object")

        session_id = _coerce_str(session.get("session_id")) or _coerce_str(
            session.get("id")
        )
        turns_raw = (
            session.get("dialogue") or session.get("turns") or session.get("messages")
        )
        turns = _parse_turns(turns_raw, context=f"conversation[{idx}]")
        out.append(LoCoMoSession(session_id=session_id, turns=turns, date_time=None))

    return tuple(out)


_SESSION_KEY_RE = re.compile(r"^session_(?P<n>\d+)$")


def _parse_conversation_dict(value: dict[str, Any]) -> tuple[LoCoMoSession, ...]:
    sessions: list[tuple[int, str]] = []
    for key, v in value.items():
        if not isinstance(key, str):
            continue
        m = _SESSION_KEY_RE.match(key)
        if not m:
            continue
        if not isinstance(v, list):
            continue
        sessions.append((int(m.group("n")), key))

    sessions.sort(key=lambda t: t[0])

    out: list[LoCoMoSession] = []
    for n, key in sessions:
        date_time = _coerce_str(value.get(f"session_{n}_date_time"))
        turns = _parse_turns(value.get(key), context=f"conversation[{key}]")
        out.append(LoCoMoSession(session_id=key, turns=turns, date_time=date_time))
    return tuple(out)


def _parse_turns(value: Any, *, context: str) -> tuple[LoCoMoTurn, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{context}.dialogue must be a list of turns")

    out: list[LoCoMoTurn] = []
    for idx, turn in enumerate(value):
        if not isinstance(turn, dict):
            raise ValueError(f"{context}.dialogue[{idx}] must be an object")

        turn_id = _coerce_str(turn.get("dia_id")) or _coerce_str(turn.get("turn_id"))
        speaker = (
            _coerce_str(turn.get("speaker")) or _coerce_str(turn.get("role")) or ""
        )
        text = _coerce_str(turn.get("text")) or _coerce_str(turn.get("content")) or ""
        if not text:
            continue

        timestamp = _coerce_str(turn.get("timestamp"))
        out.append(
            LoCoMoTurn(turn_id=turn_id, speaker=speaker, text=text, timestamp=timestamp)
        )

    return tuple(out)


def _parse_qa(value: Any) -> tuple[LoCoMoQA, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError("'qa' must be a list")

    out: list[LoCoMoQA] = []
    for idx, qa in enumerate(value):
        if not isinstance(qa, dict):
            raise ValueError(f"qa[{idx}] must be an object")

        qid = (
            _coerce_str(qa.get("question_id")) or _coerce_str(qa.get("id")) or f"q{idx}"
        )
        question = _coerce_str(qa.get("question")) or ""
        if not question:
            continue

        answers_raw = qa.get("answer") if "answer" in qa else qa.get("answers")
        answers = _coerce_answers(answers_raw)
        category = _coerce_str(qa.get("category")) or _coerce_str(qa.get("type"))
        evidence = (
            qa.get("evidence") if "evidence" in qa else qa.get("supporting_facts")
        )
        out.append(
            LoCoMoQA(
                question_id=qid,
                question=question,
                answers=answers,
                category=category,
                evidence=evidence,
            )
        )

    return tuple(out)


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value).strip() or None


def _coerce_answers(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        v = value.strip()
        return (v,) if v else ()
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            s = _coerce_str(item)
            if s:
                out.append(s)
        return tuple(out)
    s = _coerce_str(value)
    return (s,) if s else ()
