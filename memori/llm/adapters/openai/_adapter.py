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
from memori.llm._utils import agno_is_openai, llm_is_openai


@Registry.register_adapter(llm_is_openai)
@Registry.register_adapter(agno_is_openai)
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
            query = payload["conversation"]["query"]
        except KeyError:
            return []

        if "input" in query or "instructions" in query:
            messages = []
            instructions = query.get("instructions", "")
            if instructions:
                clean = instructions.split("<memori_context>")[0].strip() if "<memori_context>" in instructions else instructions
                if clean:
                    messages.append({"role": "system", "content": clean})

            input_val = query.get("input", [])
            if isinstance(input_val, str):
                messages.append({"role": "user", "content": input_val})
            elif isinstance(input_val, list):
                for item in input_val:
                    if isinstance(item, dict):
                        role, content = item.get("role", "user"), item.get("content", "")
                        if isinstance(content, str):
                            messages.append({"role": role, "content": content})
                        elif isinstance(content, list):
                            text_parts = []
                            for c in content:
                                if isinstance(c, dict) and c.get("type") == "input_text":
                                    text_parts.append(c.get("text", ""))
                                elif isinstance(c, str):
                                    text_parts.append(c)
                            text = " ".join(text_parts)
                            if text.strip():
                                messages.append({"role": role, "content": text})
            return self._exclude_injected_messages(messages, payload)

        messages = query.get("messages", [])
        return self._exclude_injected_messages(messages, payload)

    def get_formatted_response(self, payload):
        try:
            response = payload["conversation"]["response"]
        except KeyError:
            return []

        if "output" in response or "output_text" in response:
            results = []
            for item in response.get("output", []):
                if isinstance(item, dict) and item.get("type") == "message":
                    for content in item.get("content", []):
                        if isinstance(content, dict):
                            if content.get("type") == "output_text":
                                results.append({"role": "assistant", "text": content.get("text", ""), "type": "text"})
                            elif content.get("type") == "refusal":
                                results.append({"role": "assistant", "text": content.get("refusal", ""), "type": "refusal"})
            if not results and response.get("output_text"):
                results.append({"role": "assistant", "text": response.get("output_text", ""), "type": "text"})
            return results

        # Chat Completions API format
        choices = response.get("choices", None)
        results = []
        if choices is not None:
            if payload["conversation"]["query"].get("stream", None) is None:
                # Unstreamed
                # [
                #   {
                #       "finish_reason": "...",
                #       "index": ...,
                #       "logprobs": ...,
                #       "message": {
                #           "annotations": ...,
                #           "audio": ...,
                #           "content": "...",
                #           "functional_calls": ...,
                #           "parsed": ...,
                #           "refusal": ...,
                #           "role": "...",
                #           "tool_calls": ...
                #       }
                #   }
                # ]
                for choice in choices:
                    message = choice.get("message", None)
                    if message is not None:
                        content = message.get("content", None)
                        if content is not None:
                            results.append(
                                {
                                    "role": message["role"],
                                    "text": content,
                                    "type": "text",
                                }
                            )
            else:
                # Streamed
                # [
                #   {
                #       "delta": {
                #           "content": "...",
                #           "function_call": ...,
                #           "refusal": ...,
                #           "role": "...",
                #           "tool_calls": ...
                #       }
                #   }
                # ]
                content = []
                role = None
                for choice in choices:
                    delta = choice.get("delta", None)
                    if delta is not None:
                        if role is None:
                            role = delta.get("role", None)

                        text_content = delta.get("content", None)
                        if text_content is not None and len(text_content) > 0:
                            content.append(text_content)

                if len(content) > 0:
                    results.append(
                        {"role": role, "text": "".join(content), "type": "text"}
                    )

        return results
