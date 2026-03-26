import json
from typing import cast

from google.protobuf import json_format

from memori.llm.helpers.serialization import str_object_mapping


def extract_text_from_parts(parts: list) -> str:
    text_parts = []
    for part in parts:
        if isinstance(part, str):
            text_parts.append(part)
        elif isinstance(part, dict) and "text" in part:
            text_parts.append(part["text"])
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)
    return " ".join(text_parts) if text_parts else ""


def extract_from_contents(contents) -> str:
    if isinstance(contents, str):
        return contents

    if isinstance(contents, list):
        for content in reversed(contents):
            if isinstance(content, str):
                return content
            content_dict = str_object_mapping(content)
            if content_dict is not None and content_dict.get("role") == "user":
                text = extract_text_from_parts(
                    cast(list[object], content_dict.get("parts", []))
                )
                if text:
                    return text
            elif getattr(content, "role", None) == "user":
                text = extract_text_from_parts(getattr(content, "parts", []))
                if text:
                    return text

    return ""


def extract_user_query(kwargs: dict) -> str:
    if "messages" in kwargs and kwargs["messages"]:
        for msg in reversed(kwargs["messages"]):
            if msg.get("role") == "user":
                return msg.get("content", "")

    if "input" in kwargs:
        input_val = kwargs.get("input", "")
        if isinstance(input_val, str):
            return input_val
        if isinstance(input_val, list):
            for item in reversed(input_val):
                item_dict = str_object_mapping(item)
                if item_dict is not None and item_dict.get("role") == "user":
                    content = item_dict.get("content", "")
                    if isinstance(content, str):
                        return content
                    if isinstance(content, list):
                        for c in content:
                            c_dict = str_object_mapping(c)
                            if (
                                c_dict is not None
                                and c_dict.get("type") == "input_text"
                            ):
                                text = c_dict.get("text", "")
                                if isinstance(text, str):
                                    return text
                            if isinstance(c, str):
                                return c

    if "contents" in kwargs:
        result = extract_from_contents(kwargs["contents"])
        if result:
            return result

    if "request" in kwargs:
        try:
            formatted_kwargs = json.loads(
                json_format.MessageToJson(kwargs["request"].__dict__["_pb"])
            )
            if "contents" in formatted_kwargs:
                return extract_from_contents(formatted_kwargs["contents"])
        except Exception:
            pass

    return ""
