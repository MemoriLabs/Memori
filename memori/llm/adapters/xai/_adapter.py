r"""
 __  __                           _
|  \/  | ___ _ __ ___   ___  _ __(_)
| |\/| |/ _ \ '_ ` _ \ / _ \| '__| |
| |  | |  __/ | | | | | (_) | |  | |
|_|  |_|\___|_| |_| |_|\___/|_|  |_|
                 perfectam memoriam
                      memorilabs.ai
"""

from memori.llm._base import BaseLlmAdaptor
from memori.llm._registry import Registry
from memori.llm._utils import agno_is_xai, llm_is_xai


@Registry.register_adapter(llm_is_xai)
@Registry.register_adapter(agno_is_xai)
class Adapter(BaseLlmAdaptor):
    def get_formatted_query(self, payload):
        try:
            raw_messages = payload["conversation"]["query"].get("messages", [])
        except KeyError:
            return []

        messages = []
        for msg in raw_messages:
            role_str = msg.get("role", "")
            role = None

            if role_str == "ROLE_USER":
                role = "user"
            elif role_str == "ROLE_ASSISTANT":
                role = "assistant"
            elif role_str == "ROLE_SYSTEM":
                role = "system"
            elif role_str in ("user", "assistant", "system"):
                role = role_str

            if role is not None:
                content_parts = msg.get("content", [])

                if isinstance(content_parts, str):
                    messages.append({"role": role, "content": content_parts})
                elif isinstance(content_parts, list):
                    content_texts = []
                    for part in content_parts:
                        text = part.get("text", None)
                        if text is not None:
                            content_texts.append(text)

                    if len(content_texts) > 0:
                        messages.append(
                            {"role": role, "content": " ".join(content_texts)}
                        )

        return self._exclude_injected_messages(messages, payload)

    def get_formatted_response(self, payload):
        try:
            response_data = payload["conversation"]["response"]
        except (KeyError, TypeError):
            return []

        if not response_data:
            return []

        response = []

        if isinstance(response_data, list):
            content_parts = []
            role = None
            for item in response_data:
                if not isinstance(item, dict):
                    continue
                if "choices" in item:
                    for choice in item.get("choices", []):
                        if not isinstance(choice, dict):
                            continue
                        msg = choice.get("delta") or choice.get("message") or {}
                        if role is None:
                            role = msg.get("role", None)
                        text = msg.get("content", None)
                        if text:
                            content_parts.append(text)
                else:
                    if role is None:
                        role = item.get("role", None)
                    item_content = item.get("content", None)
                    if isinstance(item_content, str):
                        content_parts.append(item_content)
                    elif isinstance(item_content, list):
                        for sub in item_content:
                            if isinstance(sub, dict) and sub.get("text"):
                                content_parts.append(sub.get("text"))
                            elif isinstance(sub, str):
                                content_parts.append(sub)

            if content_parts:
                response.append(
                    {
                        "role": role or "assistant",
                        "text": "".join(content_parts),
                        "type": "text",
                    }
                )
        elif isinstance(response_data, dict):
            if "choices" in response_data:
                for choice in response_data.get("choices", []):
                    if not isinstance(choice, dict):
                        continue
                    message = choice.get("message") or choice.get("delta") or {}
                    content = message.get("content", None)
                    role = message.get("role", None)
                    if content is not None and role is not None:
                        response.append({"role": role, "text": content, "type": "text"})
            else:
                content = response_data.get("content", None)
                role = response_data.get("role", None)

                if content is not None and role is not None:
                    if isinstance(content, str):
                        response.append({"role": role, "text": content, "type": "text"})
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict):
                                text = item.get("text", None)
                                if text is not None:
                                    response.append(
                                        {"role": role, "text": text, "type": "text"}
                                    )
                            elif isinstance(item, str):
                                response.append(
                                    {"role": role, "text": item, "type": "text"}
                                )

        return response
