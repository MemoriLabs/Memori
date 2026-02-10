r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                  perfectam memoriam
                       memorilabs.ai
"""

import logging
import time

from memori._config import Config
from memori._exceptions import MemoriApiError
from memori._network import Api, ApiSubdomain
from memori.memory._writer import Writer

logger = logging.getLogger(__name__)


class Manager:
    def __init__(self, config: Config):
        self.config = config

    def execute(self, payload):
        logger.debug("Memory manager execution started")
        # Make a copy of the payload and strip out the system messages while preserving the original
        payload_stripped = payload.copy()
        payload_stripped["messages"] = [
            message
            for message in payload["messages"]
            if message.get("role") != "system"
        ]

        if self.config.hosted is True:
            self._handle_hosted(payload_stripped)
        else:
            Writer(self.config).execute(payload_stripped)
        logger.debug("Memory manager execution completed")

        return self

    def _handle_hosted(self, payload):
        api = Api(self.config, ApiSubdomain.HOSTED)
        attempts = max(1, int(getattr(self.config, "request_num_backoff", 1) or 1))
        backoff_factor = float(getattr(self.config, "request_backoff_factor", 1) or 1)

        last_error: Exception | None = None
        last_status: int | None = None

        for attempt in range(attempts):
            try:
                last_status = api.post(
                    "conversation/messages",
                    payload,
                    status_code=True,
                )
                if last_status == 201:
                    self._persist_hosted_messages_locally(payload)
                    return
                last_error = None
            except Exception as e:  # noqa: BLE001
                last_error = e

            if attempt < attempts - 1:
                time.sleep(backoff_factor * (2**attempt))

        if last_error is not None:
            raise last_error

        raise MemoriApiError(
            f"Expected 201 from hosted API but received {last_status} after {attempts} attempts"
        )

    def _ensure_cached_id(self, cache_attr: str, create_func, *create_args) -> int:
        cached_id = getattr(self.config.cache, cache_attr)
        if cached_id is None:
            cached_id = create_func(*create_args)
            if cached_id is None:
                raise RuntimeError(f"{cache_attr} is unexpectedly None")
            setattr(self.config.cache, cache_attr, cached_id)
        return cached_id

    def _persist_hosted_messages_locally(self, payload: dict) -> None:
        storage = getattr(self.config, "storage", None)
        driver = getattr(storage, "driver", None) if storage is not None else None
        if driver is None:
            return

        if self.config.entity_id is not None:
            self._ensure_cached_id(
                "entity_id",
                driver.entity.create,
                self.config.entity_id,
            )

        if self.config.process_id is not None:
            self._ensure_cached_id(
                "process_id",
                driver.process.create,
                self.config.process_id,
            )

        self._ensure_cached_id(
            "session_id",
            driver.session.create,
            self.config.session_id,
            self.config.cache.entity_id,
            self.config.cache.process_id,
        )

        self._ensure_cached_id(
            "conversation_id",
            driver.conversation.create,
            self.config.cache.session_id,
            self.config.session_timeout_minutes,
        )

        messages = payload.get("messages") if isinstance(payload, dict) else None
        if not isinstance(messages, list):
            return

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            text = message.get("text")
            if role is None or text is None:
                continue
            driver.conversation.message.create(
                self.config.cache.conversation_id,
                role,
                message.get("type"),
                str(text),
            )

        adapter = getattr(storage, "adapter", None)
        if adapter is not None:
            adapter.flush()
            adapter.commit()
