import json

from google.protobuf import json_format


def append_to_list(lst: list, context: str, parent: dict, key: str):
    if not lst:
        parent[key] = [{"text": context.lstrip("\n")}]
    elif isinstance(lst[0], dict) and "text" in lst[0]:
        lst[0]["text"] += context
    elif isinstance(lst[0], str):
        lst[0] += context
    else:
        lst.insert(0, {"text": context.lstrip("\n")})


def append_to_list_obj(config, context: str):
    lst = config.system_instruction
    if not lst:
        config.system_instruction = context.lstrip("\n")
    elif hasattr(lst[0], "text"):
        lst[0].text += context
    elif isinstance(lst[0], str):
        lst[0] += context
    else:
        config.system_instruction = context.lstrip("\n")


def append_to_content_dict(content: dict, context: str, parent: dict, key: str):
    if "parts" in content:
        parts = content.get("parts", [])
        if parts and isinstance(parts[0], dict) and "text" in parts[0]:
            parts[0]["text"] += context
        else:
            if not content.get("parts"):
                content["parts"] = []
            content["parts"].insert(0, {"text": context.lstrip("\n")})
    elif "text" in content:
        content["text"] += context
    else:
        parent[key] = context.lstrip("\n")


def append_to_part_obj(part, context: str):
    if part.text:
        part.text += context
    else:
        part.text = context.lstrip("\n")


def append_to_content_obj(content, context: str):
    if content.parts and len(content.parts) > 0 and hasattr(content.parts[0], "text"):
        if content.parts[0].text:
            content.parts[0].text += context
        else:
            content.parts[0].text = context.lstrip("\n")


def append_to_google_system_instruction_dict(config: dict, context: str):
    if "system_instruction" not in config or not config["system_instruction"]:
        config["system_instruction"] = context.lstrip("\n")
        return

    existing = config["system_instruction"]
    if isinstance(existing, str):
        config["system_instruction"] = existing + context
    elif isinstance(existing, list):
        append_to_list(existing, context, config, "system_instruction")
    elif isinstance(existing, dict):
        append_to_content_dict(existing, context, config, "system_instruction")
    else:
        config["system_instruction"] = context.lstrip("\n")


def append_to_google_system_instruction_obj(config, context: str):
    if not hasattr(config, "system_instruction"):
        return

    if config.system_instruction is None:
        config.system_instruction = context.lstrip("\n")
    elif isinstance(config.system_instruction, str):
        config.system_instruction = config.system_instruction + context
    elif isinstance(config.system_instruction, list):
        append_to_list_obj(config, context)
    elif hasattr(config.system_instruction, "text"):
        append_to_part_obj(config.system_instruction, context)
    elif hasattr(config.system_instruction, "parts"):
        append_to_content_obj(config.system_instruction, context)
    else:
        config.system_instruction = context.lstrip("\n")


def inject_google_system_instruction(kwargs: dict, context: str):
    if "request" in kwargs:
        formatted_kwargs = json.loads(
            json_format.MessageToJson(kwargs["request"].__dict__["_pb"])
        )
        system_instruction = formatted_kwargs.get("systemInstruction", {})
        parts = system_instruction.get("parts", [])
        if parts and isinstance(parts[0], dict) and "text" in parts[0]:
            parts[0]["text"] += context
        else:
            system_instruction["parts"] = [{"text": context.lstrip("\n")}]
        formatted_kwargs["systemInstruction"] = system_instruction
        json_format.ParseDict(formatted_kwargs, kwargs["request"].__dict__["_pb"])
        return

    config = kwargs.get("config", None)
    if config is None:
        kwargs["config"] = {"system_instruction": context.lstrip("\n")}
    elif isinstance(config, dict):
        append_to_google_system_instruction_dict(config, context)
    else:
        append_to_google_system_instruction_obj(config, context)
