import copy
import json
from collections.abc import Mapping
from typing import Any, cast

from google.protobuf import json_format

from memori.llm._utils import provider_is_langchain


def str_object_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping) and all(isinstance(k, str) for k in value.keys()):
        return cast(Mapping[str, object], value)
    return None


def convert_to_json(obj, _seen=None):
    if _seen is None:
        _seen = set()

    obj_id = id(obj)
    if obj_id in _seen:
        return None
    _seen.add(obj_id)

    try:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, list):
            return [convert_to_json(item, _seen.copy()) for item in obj]
        if isinstance(obj, dict):
            return {
                key: convert_to_json(value, _seen.copy())
                for key, value in obj.items()
                if not key.startswith("_")
            }
        if hasattr(obj, "model_dump"):
            try:
                return obj.model_dump()
            except Exception:
                pass
        if hasattr(obj, "__dict__"):
            filtered_dict = {
                k: v
                for k, v in obj.__dict__.items()
                if not k.startswith("_") and not callable(v)
            }
            if filtered_dict:
                return convert_to_json(filtered_dict, _seen.copy())
            return None
        return obj
    except Exception:
        return None


def dict_to_json(dict_: dict) -> dict:
    return convert_to_json(dict_)


def format_kwargs(
    kwargs, uses_protobuf: bool, framework_provider: str | None, injected_count: int
):
    if uses_protobuf:
        if "request" in kwargs:
            formatted_kwargs = json.loads(
                json_format.MessageToJson(kwargs["request"].__dict__["_pb"])
            )
        else:
            formatted_kwargs = copy.deepcopy(kwargs)
            formatted_kwargs = dict_to_json(formatted_kwargs)
    else:
        formatted_kwargs = copy.deepcopy(kwargs)
        if provider_is_langchain(framework_provider):
            if "response_format" in formatted_kwargs and isinstance(
                formatted_kwargs["response_format"], object
            ):
                del formatted_kwargs["response_format"]

        formatted_kwargs = dict_to_json(formatted_kwargs)

    if injected_count > 0:
        formatted_kwargs["_memori_injected_count"] = injected_count

    return formatted_kwargs


def safe_copy(obj):
    try:
        return copy.deepcopy(obj)
    except (TypeError, AttributeError):
        pass

    if isinstance(obj, list):
        return [safe_copy(item) for item in obj]

    if isinstance(obj, dict):
        return {key: safe_copy(value) for key, value in obj.items()}

    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return copy.copy(obj)
        except Exception:
            pass

    return obj


def format_response(raw_response, uses_protobuf: bool):
    formatted_response = safe_copy(raw_response)
    if uses_protobuf and not isinstance(formatted_response, list):
        if (
            hasattr(formatted_response, "__dict__")
            and "_pb" in formatted_response.__dict__
        ):
            return json.loads(
                json_format.MessageToJson(formatted_response.__dict__["_pb"])
            )
        if hasattr(formatted_response, "candidates"):
            result: dict[str, Any] = {}
            if formatted_response.candidates:
                candidates = []
                for candidate in formatted_response.candidates:
                    candidate_data = {}
                    if hasattr(candidate, "content") and candidate.content:
                        content_data = {}
                        if (
                            hasattr(candidate.content, "parts")
                            and candidate.content.parts
                        ):
                            parts = []
                            for part in candidate.content.parts:
                                if hasattr(part, "text"):
                                    parts.append({"text": part.text})
                            content_data["parts"] = parts
                        if hasattr(candidate.content, "role"):
                            content_data["role"] = candidate.content.role
                        candidate_data["content"] = content_data
                    candidates.append(candidate_data)
                result["candidates"] = candidates
            return result
        return {}

    return formatted_response


def get_response_content(raw_response):
    if (
        raw_response.__class__.__name__ == "LegacyAPIResponse"
        and raw_response.__class__.__module__ == "openai._legacy_response"
    ):
        return json.loads(raw_response.text)

    if hasattr(raw_response, "output") and hasattr(raw_response, "output_text"):
        if hasattr(raw_response, "model_dump"):
            return raw_response.model_dump()
        if hasattr(raw_response, "__dict__"):
            return convert_to_json(raw_response)

    return raw_response
