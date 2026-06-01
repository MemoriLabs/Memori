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
from memori.llm._utils import llm_is_bedrock


@Registry.register_adapter(llm_is_bedrock)
class Adapter(BaseLlmAdaptor):
    def get_formatted_query(self, payload):
        """
        [
            {
                "content": "...",
                "role": "..."
            }
        ]
        """

        try:
            messages = payload["conversation"]["query"]["body"]["messages"]
            return self._exclude_injected_messages(messages, payload)
        except KeyError:
            return []

    def get_formatted_response(self, payload):
        try:
            if not isinstance(payload["conversation"]["response"], list):
                return []
        except KeyError:
            return []

        response = []
        if "chunk" not in payload["conversation"]["response"][0]:
            # Unstreamed
            data = payload["conversation"]["response"][0]
            role = None
            text = []

            # Converse API format
            if "output" in data and "message" in data["output"]:
                message = data["output"]["message"]
                role = message.get("role")
                content = message.get("content", [])
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        text.append(part["text"])

            # Messages API format (e.g. Claude 3 directly on Bedrock)
            elif "role" in data and "content" in data:
                role = data["role"]
                content = data["content"]
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            text.append(part["text"])
                        elif isinstance(part, str):
                            text.append(part)
                elif isinstance(content, str):
                    text.append(content)

            # Legacy Claude on Bedrock format
            elif "completion" in data:
                role = "assistant"
                text.append(data["completion"])

            if role and text:
                response.append({"role": role, "text": "".join(text), "type": "text"})
        else:
            # Streamed
            # [
            #   {
            #       "chunk": {
            #           "bytes": {
            #               "delta": {
            #                   "text": "...",
            #                   "type": "..."
            #               }
            #           }
            #       }
            #   }
            # ]
            response = []
            text = []
            role = None
            for entry in payload["conversation"]["response"]:
                chunk = entry.get("chunk", None)
                if chunk is not None:
                    bytes_ = chunk.get("bytes", None)
                    if bytes_ is not None:
                        message = bytes_.get("message", None)
                        if message is not None:
                            role = message["role"]
                        else:
                            delta = bytes_.get("delta", None)
                            if delta is not None:
                                text_content = delta.get("text", None)
                                if text_content is not None and len(text_content) > 0:
                                    text.append(text_content)

            if len(text) > 0:
                response.append({"role": role, "text": "".join(text), "type": "text"})

        return response
