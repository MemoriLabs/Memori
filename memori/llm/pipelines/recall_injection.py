import logging
from collections.abc import Mapping
from typing import cast

from memori._logging import truncate
from memori._utils import format_date_created
from memori.llm._utils import (
    agno_is_google,
    llm_is_anthropic,
    llm_is_bedrock,
    llm_is_google,
)
from memori.llm.helpers.google_system_instruction import (
    inject_google_system_instruction,
)
from memori.llm.helpers.query_extraction import extract_user_query
from memori.memory.recall import (
    _collect_cloud_summaries_from_facts,
    _score_for_recall_threshold,
)
from memori.search._types import FactSearchResult

logger = logging.getLogger(__name__)


def format_recalled_fact_lines(
    facts: list[FactSearchResult | Mapping[str, object] | str],
) -> list[str]:
    lines: list[str] = []
    for fact in facts:
        if isinstance(fact, str):
            if fact:
                lines.append(f"- {fact}")
            continue
        if isinstance(fact, Mapping):
            fact_map = cast(Mapping[str, object], fact)
            content = fact_map.get("content")
            date_created = fact_map.get("date_created")
        elif hasattr(fact, "content") and hasattr(fact, "date_created"):
            content = fact.content
            date_created = fact.date_created
        else:
            continue

        if not content:
            continue
        ts = format_date_created(date_created)
        suffix = f". Stated at {ts}" if ts else ""
        lines.append(f"- {content}{suffix}")
    return lines


def format_recalled_summary_lines(summaries: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    seen: set[tuple[str, str]] = set()
    for summary in summaries:
        content = summary.get("content")
        if not isinstance(content, str) or not content:
            continue

        date_created = summary.get("date_created")
        date_key = date_created if isinstance(date_created, str) else ""
        summary_key = (content, date_key)
        if summary_key in seen:
            continue
        seen.add(summary_key)

        ts = format_date_created(summary.get("date_created"))
        if ts:
            lines.append(f"- [{ts}]\n  {content}")
        else:
            lines.append(f"- {content}")
    return lines


def inject_recalled_facts(invoke, kwargs: dict) -> dict:
    invoke._cloud_summaries = []
    if invoke.config.cloud is True:
        invoke._cloud_conversation_messages = []

    if invoke.config.entity_id is None:
        return kwargs

    user_query = extract_user_query(kwargs)
    if not user_query:
        return kwargs

    logger.debug("User query: %s", truncate(user_query))

    resolved_entity_id = None
    if invoke.config.cloud is False:
        if invoke.config.storage is None or invoke.config.storage.driver is None:
            return kwargs

        resolved_entity_id = invoke.config.storage.driver.entity.create(
            invoke.config.entity_id
        )
        if resolved_entity_id is None:
            return kwargs

    from memori.memory.recall import Recall

    recall = Recall(invoke.config)
    if invoke.config.cloud is True:
        from memori.memory.recall import CloudRecallResponse

        cloud_response = cast(CloudRecallResponse, recall.search_facts(user_query))
        facts = cloud_response["facts"]
        invoke._cloud_conversation_messages = cloud_response.get("messages", [])
        invoke._cloud_summaries = _collect_cloud_summaries_from_facts(facts)
    else:
        facts = cast(
            list[FactSearchResult | Mapping[str, object] | str],
            recall.search_facts(
                user_query,
                entity_id=resolved_entity_id,
                cloud=bool(invoke.config.cloud),
            ),
        )
        invoke._cloud_summaries = _collect_cloud_summaries_from_facts(facts)

    if not facts:
        logger.debug("No facts found to inject into prompt")
        return kwargs

    relevant_facts = [
        f
        for f in facts
        if _score_for_recall_threshold(f) >= invoke.config.recall_relevance_threshold
    ]

    if not relevant_facts:
        logger.debug(
            "No facts above relevance threshold (%.2f)",
            invoke.config.recall_relevance_threshold,
        )
        return kwargs

    logger.debug("Injecting %d recalled facts into prompt", len(relevant_facts))

    # Keep summaries scoped to the same relevant fact subset we inject.
    invoke._cloud_summaries = _collect_cloud_summaries_from_facts(
        cast(list, relevant_facts)
    )
    fact_lines = format_recalled_fact_lines(relevant_facts)
    summary_lines = format_recalled_summary_lines(invoke._cloud_summaries)
    context_body = "Relevant context about the user:\n" + "\n".join(fact_lines)
    if summary_lines:
        context_body += "\n\n## Summaries\n\n" + "\n\n".join(summary_lines)
    recall_context = (
        "\n\n<memori_context>\n"
        "Only use the relevant context if it is relevant to the user's query. "
        + context_body
        + "\n</memori_context>"
    )
    print(recall_context)
    if llm_is_anthropic(
        invoke.config.framework.provider, invoke.config.llm.provider
    ) or llm_is_bedrock(invoke.config.framework.provider, invoke.config.llm.provider):
        existing_system = kwargs.get("system", "")
        kwargs["system"] = existing_system + recall_context
    elif llm_is_google(
        invoke.config.framework.provider, invoke.config.llm.provider
    ) or agno_is_google(invoke.config.framework.provider, invoke.config.llm.provider):
        inject_google_system_instruction(kwargs, recall_context)
    elif ("input" in kwargs or "instructions" in kwargs) and "messages" not in kwargs:
        existing_instructions = kwargs.get("instructions", "") or ""
        kwargs["instructions"] = existing_instructions + recall_context
    else:
        messages = kwargs.get("messages", [])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = messages[0]["content"] + recall_context
        else:
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": recall_context.lstrip("\n"),
                },
            )

    return kwargs
