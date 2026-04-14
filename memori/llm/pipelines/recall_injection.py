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


def format_recalled_summary_lines(
    facts: list[FactSearchResult | Mapping[str, object] | str],
) -> list[str]:
    primary_summaries: list[dict[str, object]] = []
    additional_summaries: list[dict[str, object]] = []

    for fact in facts:
        summaries_raw: object | None = None
        if isinstance(fact, Mapping):
            fact_map = cast(Mapping[str, object], fact)
            summaries_raw = fact_map.get("summaries")
        elif hasattr(fact, "summaries"):
            summaries_raw = fact.summaries

        if not isinstance(summaries_raw, list):
            continue

        has_primary_for_fact = False
        for summary in summaries_raw:
            if not isinstance(summary, Mapping):
                continue
            summary_map = cast(Mapping[str, object], summary)
            content = summary_map.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            summary_dict = {
                "content": content,
                "date_created": summary_map.get("date_created"),
            }
            if not has_primary_for_fact:
                primary_summaries.append(summary_dict)
                has_primary_for_fact = True
            else:
                additional_summaries.append(summary_dict)

    lines: list[str] = []
    seen: set[str] = set()
    for summary_group in (primary_summaries, additional_summaries):
        for summary in summary_group:
            content = cast(str, summary["content"])
            content_key = content.strip()
            if content_key in seen:
                continue
            seen.add(content_key)

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

    if invoke.config.cloud is True:
        from memori.memory.recall import CloudRecallResponse, Recall

        recall = Recall(invoke.config)
        cloud_response = cast(CloudRecallResponse, recall.search_facts(user_query))
        facts = cloud_response["facts"]
        invoke._cloud_conversation_messages = cloud_response.get("messages", [])
        invoke._cloud_summaries = _collect_cloud_summaries_from_facts(facts)
    else:
        rust_core = getattr(invoke.config, "rust_core", None)
        if rust_core is not None:
            try:
                facts = cast(
                    list[FactSearchResult | Mapping[str, object] | str],
                    rust_core.retrieve_facts(
                        query=user_query,
                        entity_id=str(resolved_entity_id),
                        limit=invoke.config.recall_facts_limit,
                        dense_limit=invoke.config.recall_embeddings_limit,
                    ),
                )
            except Exception:
                from memori.memory.recall import Recall

                recall = Recall(invoke.config)
                facts = cast(
                    list[FactSearchResult | Mapping[str, object] | str],
                    recall.search_facts(
                        user_query,
                        entity_id=resolved_entity_id,
                        cloud=bool(invoke.config.cloud),
                    ),
                )
        else:
            from memori.memory.recall import Recall

            recall = Recall(invoke.config)
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
    summary_lines = format_recalled_summary_lines(relevant_facts)
    context_body = "Relevant context about the user:\n" + "\n".join(fact_lines)
    if summary_lines:
        context_body += "\n\n## Summaries\n\n" + "\n\n".join(summary_lines)
    recall_context = (
        "\n\n<memori_context>\n"
        "Only use the relevant context if it is relevant to the user's query. "
        + context_body
        + "\n</memori_context>"
    )

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
