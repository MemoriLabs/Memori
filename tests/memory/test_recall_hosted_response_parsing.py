from __future__ import annotations

from memori._config import Config
from memori.memory.recall import Recall


def test_hosted_recall_caches_conversation_messages_and_returns_facts(mocker):
    cfg = Config()
    cfg.hosted = True
    cfg.entity_id = "entity-1"
    cfg.process_id = "process-1"
    cfg.session_id = "session-1"

    api_instance = mocker.MagicMock()
    api_instance.post.return_value = {
        "conversation": {
            "messages": [
                {"role": "user", "text": "Hi", "type": "message"},
                {"role": "assistant", "text": "Hello", "type": "message"},
            ]
        },
        "facts": [
            {
                "id": 1,
                "content": "User likes pizza",
                "date_created": "2026-02-01T00:00:00Z",
                "rank_score": 0.9,
                "similarity": 0.8,
            }
        ],
    }
    mock_api_cls = mocker.patch("memori.memory.recall.Api", autospec=True)
    mock_api_cls.return_value = api_instance

    facts = Recall(cfg).search_facts(query="What do I like?", limit=5)

    assert facts == api_instance.post.return_value["facts"]
    assert cfg.cache.hosted_conversation_messages == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
