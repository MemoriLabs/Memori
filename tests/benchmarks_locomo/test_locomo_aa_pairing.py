from __future__ import annotations

from benchmarks.locomo._run_impl import _build_per_pair_requests


def test_build_per_pair_requests_cumulative() -> None:
    msgs = [
        {"role": "user", "content": "u0"},
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    turn_ids = ["D1:1", "D1:2", "D1:3", "D1:4"]
    reqs = _build_per_pair_requests(msgs, turn_ids)
    assert len(reqs) == 2
    assert [m["content"] for m in reqs[0].messages] == ["u0", "a0"]
    assert reqs[0].pair_turn_ids == ("D1:1", "D1:2")
    assert [m["content"] for m in reqs[1].messages] == ["u0", "a0", "u1", "a1"]
    assert reqs[1].pair_turn_ids == ("D1:3", "D1:4")


def test_build_per_pair_requests_skips_unpairable_pair_boundaries() -> None:
    msgs = [
        {"role": "assistant", "content": "a0"},
        {"role": "user", "content": "u0"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    turn_ids = ["D1:1", "D1:2", "D1:3", "D1:4", "D1:5"]
    reqs = _build_per_pair_requests(msgs, turn_ids)
    assert len(reqs) == 1
    # Context includes everything up to the paired assistant message.
    assert [m["content"] for m in reqs[0].messages] == ["a0", "u0", "u1", "a1"]
    assert reqs[0].pair_turn_ids == ("D1:3", "D1:4")
