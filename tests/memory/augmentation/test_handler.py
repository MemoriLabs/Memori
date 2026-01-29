from memori._config import Config
from memori.memory.augmentation._handler import handle_augmentation
from memori.memory.augmentation.augmentations.memori.models import (
    AttributionData,
    AugmentationInputData,
    ConversationMessage,
    EntityData,
    ProcessData,
    SessionData,
)


def test_handle_augmentation_hosted_posts_hosted_payload(mocker):
    cfg = Config()
    cfg.hosted = True
    cfg.entity_id = "abc"
    cfg.process_id = "def"
    cfg.framework.provider = "agno"
    cfg.llm.provider = "openai"
    cfg.llm.version = "gpt-4o-mini"
    cfg.platform.provider = "local"
    cfg.request_num_backoff = 2
    cfg.request_backoff_factor = 0
    cfg.request_secs_timeout = 1
    cfg.thread_pool_executor = mocker.Mock()

    api = mocker.Mock()
    api.post.return_value = 204
    mocker.patch("memori.memory.augmentation._handler.Api", return_value=api)

    sleep = mocker.patch("memori.memory.augmentation._handler.time.sleep")

    handle_augmentation(
        config=cfg,
        payload=AugmentationInputData(
            attribution=AttributionData(
                entity=EntityData(id="abc"),
                process=ProcessData(id="def"),
            ),
            messages=[
                ConversationMessage(role="user", content="hello"),
                ConversationMessage(role="assistant", content="ok"),
            ],
            session=SessionData(id=str(cfg.session_id)),
        ),
        kwargs={},
        augmentation_manager=mocker.Mock(),
        log_content=None,
    )

    cfg.thread_pool_executor.submit.assert_called_once()
    fn, cfg_arg, payload_arg = cfg.thread_pool_executor.submit.call_args.args
    fn(cfg_arg, payload_arg)

    assert sleep.call_count == 0
    assert api.post.call_count == 1
    route, sent = api.post.call_args.args
    assert route == "hosted/augmentation"
    assert "conversation" in sent and "messages" in sent["conversation"]
    assert [m["role"] for m in sent["conversation"]["messages"]] == [
        "user",
        "assistant",
    ]
    assert sent["meta"]["framework"]["provider"] == "agno"
    assert sent["meta"]["llm"]["model"]["provider"] == "openai"
    assert sent["meta"]["platform"]["provider"] == "local"
    assert sent["meta"]["sdk"]
    assert sent["meta"]["storage"] is None
    assert sent["conversation"]["summary"] is None


def test_handle_augmentation_non_hosted_enqueues(mocker):
    cfg = Config()
    cfg.hosted = False
    cfg.entity_id = "abc"
    cfg.process_id = "def"
    cfg.cache.conversation_id = 123

    aug = mocker.Mock()

    handle_augmentation(
        config=cfg,
        payload=AugmentationInputData(
            attribution=AttributionData(
                entity=EntityData(id="abc"),
                process=ProcessData(id="def"),
            ),
            messages=[ConversationMessage(role="user", content="hi")],
            session=SessionData(id=str(cfg.session_id)),
        ),
        kwargs={},
        augmentation_manager=aug,
        log_content=None,
    )

    aug.enqueue.assert_called_once()
    input_data = aug.enqueue.call_args.args[0]
    assert input_data.conversation_id == 123
    assert input_data.entity_id == "abc"
    assert input_data.process_id == "def"
    assert input_data.conversation_messages[0].role == "user"
    assert input_data.conversation_messages[0].content == "hi"


def test_handle_augmentation_hosted_logs_error_on_failed_post(mocker):
    cfg = Config()
    cfg.hosted = True
    cfg.entity_id = "abc"
    cfg.process_id = "def"
    cfg.thread_pool_executor = mocker.Mock()

    api = mocker.Mock()
    api.post.return_value = 500
    mocker.patch("memori.memory.augmentation._handler.Api", return_value=api)
    err = mocker.patch("memori.memory.augmentation._handler.logger.error")

    handle_augmentation(
        config=cfg,
        payload=AugmentationInputData(
            attribution=AttributionData(
                entity=EntityData(id="abc"),
                process=ProcessData(id="def"),
            ),
            messages=[
                ConversationMessage(role="user", content="hello"),
                ConversationMessage(role="assistant", content="ok"),
            ],
            session=SessionData(id=str(cfg.session_id)),
        ),
        kwargs={},
        augmentation_manager=mocker.Mock(),
        log_content=None,
    )

    fn, cfg_arg, payload_arg = cfg.thread_pool_executor.submit.call_args.args
    fn(cfg_arg, payload_arg)
    assert err.called


def test_handle_augmentation_no_attribution_noops(mocker):
    cfg = Config()
    cfg.hosted = True
    cfg.entity_id = None
    cfg.process_id = None
    cfg.thread_pool_executor = mocker.Mock()

    api = mocker.Mock()
    mocker.patch("memori.memory.augmentation._handler.Api", return_value=api)

    handle_augmentation(
        config=cfg,
        payload=AugmentationInputData(
            attribution=AttributionData(
                entity=EntityData(id=None),
                process=ProcessData(id=None),
            ),
            messages=[ConversationMessage(role="user", content="hello")],
            session=SessionData(id=str(cfg.session_id)),
        ),
        kwargs={},
        augmentation_manager=mocker.Mock(),
        log_content=None,
    )

    cfg.thread_pool_executor.submit.assert_not_called()
    api.post.assert_not_called()


def test_handle_augmentation_hosted_without_executor_posts_inline(mocker):
    cfg = Config()
    cfg.hosted = True
    cfg.entity_id = "abc"
    cfg.process_id = "def"
    cfg.thread_pool_executor = None

    api = mocker.Mock()
    api.post.return_value = 204
    mocker.patch("memori.memory.augmentation._handler.Api", return_value=api)

    handle_augmentation(
        config=cfg,
        payload=AugmentationInputData(
            attribution=AttributionData(
                entity=EntityData(id="abc"),
                process=ProcessData(id="def"),
            ),
            messages=[ConversationMessage(role="user", content="hello")],
            session=SessionData(id=str(cfg.session_id)),
        ),
        kwargs={},
        augmentation_manager=mocker.Mock(),
        log_content=None,
    )

    api.post.assert_called_once()
