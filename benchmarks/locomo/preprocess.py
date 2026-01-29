from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_SESSION_KEY_RE = re.compile(r"^session_(?P<n>\d+)$")
_MULTIMODAL_KEYS = ("img_url", "blip_caption", "query")


def preprocess_locomo_json(in_path: str | Path, out_path: str | Path) -> dict[str, int]:
    """
    Preprocess LoCoMo into a Memori-friendly role format.

    - Removes multimodal turn fields (img_url, blip_caption, query).
    - Rewrites turn speakers so conversation.speaker_b becomes "assistant" and all others "user".
      If conversation.speaker_b is missing, the sample is kept and left unchanged.
    """
    src = Path(in_path)
    dst = Path(out_path)

    data = json.loads(src.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("LoCoMo JSON must be a list of samples")

    kept: list[dict[str, Any]] = []
    dropped_reasons: Counter[str] = Counter()
    removed_multimodal_turns = 0

    for _idx, sample in enumerate(data):
        if not isinstance(sample, dict):
            dropped_reasons["invalid_sample"] += 1
            continue

        conv = sample.get("conversation")
        removed_multimodal_turns += _strip_multimodal_fields_inplace(conv)

        speaker_b = None
        if isinstance(conv, dict):
            speaker_b = _coerce_str(conv.get("speaker_b"))

        if speaker_b:
            _rewrite_speakers_inplace(conv, assistant_speaker=speaker_b)
        else:
            # Keep unchanged when we can't confidently map assistant roles.
            dropped_reasons["missing_speaker_b"] += 0

        kept.append(sample)

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")

    return {
        "samples_in": len(data),
        "samples_out": len(kept),
        "removed_multimodal_turns": int(removed_multimodal_turns),
        "dropped_invalid_sample": int(dropped_reasons.get("invalid_sample", 0)),
    }


def _strip_multimodal_fields_inplace(conversation: Any) -> int:
    removed = 0
    for turn in _iter_turn_dicts(conversation):
        if not isinstance(turn, dict):
            continue
        before = len(turn)
        for k in _MULTIMODAL_KEYS:
            turn.pop(k, None)
        removed += int(len(turn) != before)
    return removed


def _rewrite_speakers_inplace(conversation: Any, *, assistant_speaker: str) -> None:
    for turn in _iter_turn_dicts(conversation):
        speaker = _coerce_str(turn.get("speaker")) or ""
        turn["speaker"] = "assistant" if speaker == assistant_speaker else "user"


def _iter_turn_dicts(conversation: Any):
    if conversation is None:
        return

    if isinstance(conversation, list):
        for session in conversation:
            if not isinstance(session, dict):
                continue
            turns = (
                session.get("dialogue")
                or session.get("turns")
                or session.get("messages")
            )
            if not isinstance(turns, list):
                continue
            for t in turns:
                if isinstance(t, dict):
                    yield t
        return

    if isinstance(conversation, dict):
        sessions: list[tuple[int, str]] = []
        for key, v in conversation.items():
            if not isinstance(key, str):
                continue
            m = _SESSION_KEY_RE.match(key)
            if not m:
                continue
            if isinstance(v, list):
                sessions.append((int(m.group("n")), key))
        sessions.sort(key=lambda x: x[0])

        for _, key in sessions:
            v = conversation.get(key)
            if not isinstance(v, list):
                continue
            for t in v:
                if isinstance(t, dict):
                    yield t
        return


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip()
        return v or None
    return str(value).strip() or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Preprocess LoCoMo: strip multimodal fields and map speaker_b -> assistant."
    )
    parser.add_argument(
        "--in", dest="in_path", required=True, help="Input LoCoMo JSON path."
    )
    parser.add_argument(
        "--out", dest="out_path", required=True, help="Output JSON path."
    )
    args = parser.parse_args(argv)

    stats = preprocess_locomo_json(args.in_path, args.out_path)
    print(
        "[locomo][preprocess] "
        f"samples_in={stats['samples_in']} samples_out={stats['samples_out']} "
        f"removed_multimodal_turns={stats['removed_multimodal_turns']} "
        f"dropped_invalid_sample={stats['dropped_invalid_sample']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
