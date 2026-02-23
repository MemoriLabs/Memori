import logging
import time
from collections.abc import Callable
from dataclasses import asdict
from typing import Any

from memori._exceptions import MemoriApiError
from memori._network import Api, ApiSubdomain
from memori.memory.augmentation.augmentations.memori.models import AugmentationInputData
from memori.memory.augmentation.input import AugmentationInput

logger = logging.getLogger(__name__)


def _build_meta(config) -> dict[str, object]:
    sdk = {
        "lang": "python",
        "version": getattr(config, "version", None),
    }

    framework = getattr(getattr(config, "framework", None), "provider", None)
    framework = {"provider": framework} if framework else None

    llm = {
        "model": {
            "provider": getattr(getattr(config, "llm", None), "provider", None),
            "sdk": {
                "version": getattr(
                    getattr(config, "llm", None), "provider_sdk_version", None
                ),
            },
            "version": getattr(getattr(config, "llm", None), "version", None),
        },
    }

    provider = getattr(getattr(config, "platform", None), "provider", None)
    platform = {"provider": provider} if provider else None

    if config.cloud is True:
        storage = None
    else:
        storage = {
            "cockroachdb": getattr(
                getattr(config, "storage_config", None), "cockroachdb", None
            ),
            "dialect": getattr(
                getattr(config, "storage_config", None), "dialect", None
            ),
        }

    return {
        "attribution": {
            "entity": {"id": config.entity_id},
            "process": {"id": config.process_id},
        },
        "framework": framework,
        "llm": llm,
        "platform": platform,
        "sdk": sdk,
        "storage": storage,
    }


def _post_cloud_augmentation(config, payload: dict) -> None:
    api = Api(config, ApiSubdomain.COLLECTOR)
    attempts = max(1, int(getattr(config, "request_num_backoff", 1) or 1))
    backoff_factor = float(getattr(config, "request_backoff_factor", 1) or 1)

    last_error: Exception | None = None
    last_status: int | None = None

    for attempt in range(attempts):
        try:
            last_status = api.post("cloud/augmentation", payload, status_code=True)
            if 200 <= last_status <= 299:
                return
            last_error = None
        except Exception as e:  # noqa: BLE001
            last_error = e

        if attempt < attempts - 1:
            time.sleep(backoff_factor * (2**attempt))

    if last_error is not None:
        raise last_error

    raise MemoriApiError(
        f"cloud augmentation request failed (status={last_status}) after {attempts} attempts"
    )


def _send_cloud_augmentation_background(config, payload: dict) -> None:
    try:
        _post_cloud_augmentation(config, payload)
    except Exception as e:  # noqa: BLE001
        logger.error("cloud augmentation background task failed: %s", e, exc_info=True)


def handle_augmentation(
    *,
    config,
    payload: AugmentationInputData,
    kwargs: dict,
    augmentation_manager: Any,
    log_content: Callable[[str], None] | None = None,
) -> None:
    if not config.entity_id and not config.process_id:
        return

    payload_dict = asdict(payload)
    if config.cloud is True:
        aug_payload = {
            "conversation": {
                "messages": payload_dict["messages"],
                "summary": None,
            },
            "meta": _build_meta(config),
            "session": {"id": str(config.session_id)},
        }

        executor = getattr(config, "thread_pool_executor", None)
        if executor is not None:
            executor.submit(_send_cloud_augmentation_background, config, aug_payload)
        else:
            _send_cloud_augmentation_background(config, aug_payload)
        return

    augmentation_input = AugmentationInput(
        conversation_id=config.cache.conversation_id,
        entity_id=config.entity_id,
        process_id=config.process_id,
        conversation_messages=payload.messages,
    )
    augmentation_manager.enqueue(augmentation_input)
