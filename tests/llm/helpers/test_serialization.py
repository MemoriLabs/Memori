from memori.llm.helpers.serialization import convert_to_json

# --- convert_to_json behavioral contracts ---


def test_convert_to_json_excludes_private_keys():
    """
    Behavioral Guarantee: convert_to_json strips all dict keys prefixed
    with '_', preventing internal SDK state from leaking into ingestion
    payloads.

    This contract is architecturally critical: format_kwargs relies on it
    to ensure that metadata like _memori_injected_count (added
    post-serialization) is the only _-prefixed key reaching the ingestion
    API. If this filter regresses, internal Python state and Memori metadata
    silently pollute stored conversation data.
    """
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4",
        "_internal_state": "should be stripped",
        "_memori_injected_count": 2,
    }

    result = convert_to_json(payload)

    assert result == {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4",
    }


def test_convert_to_json_handles_circular_references():
    """
    Behavioral Guarantee: convert_to_json breaks infinite recursion loops
    by returning None when encountering an object it has already seen.

    This is a critical crash-prevention contract. LLM SDK response objects
    often contain cyclic back-references (e.g., response.request.response).
    If this recursion guard is lost, ingestion serialization will hit a
    RecursionError, crashing the worker process and causing total data loss
    for that turn.
    """

    class CyclicNode:
        def __init__(self):
            self.value = "data"
            self.next = None

    node_a = CyclicNode()
    node_b = CyclicNode()

    # Create a circular reference: A -> B -> A
    node_a.next = node_b
    node_b.next = node_a

    result = convert_to_json(node_a)

    # node_a resolves to dict containing 'value' and 'next'.
    # 'next' resolves to node_b, which is a dict containing 'value' and 'next'.
    # node_b's 'next' points to node_a, which has already been seen (returns None).
    assert result == {
        "value": "data",
        "next": {
            "value": "data",
            "next": None,
        },
    }


def test_convert_to_json_handles_datetime_and_date():
    """
    Behavioral Guarantee: convert_to_json safely serializes datetime and date
    objects into standard ISO 8601 strings.
    """
    from datetime import date, datetime

    from memori.llm.helpers.serialization import convert_to_json

    dt = datetime(2025, 1, 1, 15, 30, 45)
    d = date(2025, 1, 1)

    payload = {"timestamp": dt, "day": d, "nested": {"time": dt}}

    result = convert_to_json(payload)

    assert result == {
        "timestamp": "2025-01-01T15:30:45",
        "day": "2025-01-01",
        "nested": {"time": "2025-01-01T15:30:45"},
    }


# --- safe_copy behavioral contracts ---


def test_safe_copy_isolates_mutation_for_uncopyable_dict_values():
    """
    Behavioral Guarantee: safe_copy creates a new dictionary even when
    values cannot be deepcopied, falling back to preserving the uncopyable
    values by reference.

    This protects against two ingestion failures:
    1. Crashing the pipeline when a provider SDK returns an uncopyable object
       (e.g., an httpx.Response socket in an OpenAI payload).
    2. Mutating the original response object during downstream normalization,
       which could cause unexpected side effects in the user's application.
    """
    from memori.llm.helpers.serialization import safe_copy

    class Uncopyable:
        def __deepcopy__(self, memo):
            raise TypeError("Cannot deepcopy")

        def __copy__(self):
            raise TypeError("Cannot shallow copy")

    uncopyable = Uncopyable()
    original = {"data": "value", "sdk_obj": uncopyable}

    result = safe_copy(original)

    # 1. Pipeline does not crash and returns a valid dict
    assert isinstance(result, dict)
    assert result["data"] == "value"

    # 2. Uncopyable object is preserved by reference (defensive fallback)
    assert result["sdk_obj"] is uncopyable

    # 3. Mutation isolation: changing result does not alter original
    result["data"] = "mutated"
    assert original["data"] == "value"


# --- format_kwargs behavioral contracts ---


def test_format_kwargs_injects_memori_injected_count():
    """
    Behavioral Guarantee: format_kwargs injects the _memori_injected_count
    metadata field into the serialized payload when injected_count > 0.

    This is a critical cross-function ingestion contract. Downstream
    provider adapters (BaseLlmAdaptor._exclude_injected_messages) rely on
    this exact metadata key to slice out injected RAG context before storing
    the conversation history. If this injection fails or the key changes,
    the conversation history will permanently store injected system prompts,
    leading to exponential token bloat and context corruption.
    """
    from memori.llm.helpers.serialization import format_kwargs

    kwargs = {"temperature": 0.7, "model": "gpt-4"}

    # 1. When injected_count > 0, metadata is appended to the serialized dict
    result_with_injection = format_kwargs(
        kwargs=kwargs,
        uses_protobuf=False,
        framework_provider=None,
        injected_count=3,
    )

    assert result_with_injection["_memori_injected_count"] == 3
    assert result_with_injection["temperature"] == 0.7

    # 2. When injected_count is 0, metadata is NOT appended (avoids schema bloat)
    result_no_injection = format_kwargs(
        kwargs=kwargs,
        uses_protobuf=False,
        framework_provider=None,
        injected_count=0,
    )

    assert "_memori_injected_count" not in result_no_injection


# --- get_response_content behavioral contracts ---


def test_get_response_content_unknown_type_passthrough():
    """
    Behavioral Guarantee: get_response_content returns unknown or standard
    response objects completely unchanged, acting as a safe passthrough.

    This is a critical defensive fallback contract. While the function unpacks
    specific legacy or wrapper formats (e.g., OpenAI LegacyAPIResponse), the
    vast majority of provider responses (Anthropic, Bedrock, xAI, standard OpenAI)
    must flow through here untouched. If this passthrough fails (e.g., returning
    None or mutating the object), the entire response payload for those providers
    would be silently destroyed before reaching the ingestion pipeline.
    """
    from memori.llm.helpers.serialization import get_response_content

    # A standard object that doesn't match any special legacy/wrapper signatures
    class StandardResponse:
        def __init__(self):
            self.content = "Normal response data"
            self.role = "assistant"

    raw_response = StandardResponse()

    result = get_response_content(raw_response)

    # The exact same object reference must be returned unmodified
    assert result is raw_response
    assert result.content == "Normal response data"
