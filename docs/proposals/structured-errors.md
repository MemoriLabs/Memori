# Proposal: Structured Error Types with Actionable Messages

**Author:** Developer Relations
**Status:** Proposal
**Created:** January 2026

## Summary

This proposal recommends implementing a structured exception hierarchy that provides developers with actionable guidance when errors occur. Instead of generic `RuntimeError` messages, errors would include specific workarounds, documentation links, and diagnostic steps.

## Problem Statement

### Current Behavior

Memori uses generic exception types with minimal context:

```python
# Current error (from Issue #234)
RuntimeError: Unsupported LLM client type: langchain_openai.chat_models.base.ChatOpenAI
```

### The Developer Experience

When a developer encounters this error:

1. They search Google/GitHub for the error message
2. Find Issue #234 (if lucky) or open a new issue
3. Wait for maintainer response
4. Finally get the workaround

**Result:** Hours to days of blocked development time.

### Evidence from GitHub Issues

| Issue | Error Type | Resolution | Wait Time |
|-------|-----------|------------|-----------|
| #234 | `RuntimeError: Unsupported LLM client type` | Register base client | 5 comments, 9 days |
| #238 | Silent failure (empty database) | Call `augmentation.wait()` | 1 comment, ongoing |

**Pattern:** Developers are blocked by problems that have known solutions, but the SDK doesn't communicate those solutions.

## Proposed Solution

### Exception Hierarchy

```python
# memori/exceptions.py

class MemoriError(Exception):
    """Base exception for all Memori errors."""

    def __init__(
        self,
        message: str,
        docs_url: str = None,
        suggestions: list[str] = None,
        diagnostic_steps: list[str] = None
    ):
        self.message = message
        self.docs_url = docs_url or "https://memorilabs.ai/docs"
        self.suggestions = suggestions or []
        self.diagnostic_steps = diagnostic_steps or []
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        parts = [self.message]

        if self.suggestions:
            parts.append("\nSuggestions:")
            for i, suggestion in enumerate(self.suggestions, 1):
                parts.append(f"  {i}. {suggestion}")

        if self.diagnostic_steps:
            parts.append("\nDiagnostic steps:")
            for i, step in enumerate(self.diagnostic_steps, 1):
                parts.append(f"  {i}. {step}")

        parts.append(f"\nDocumentation: {self.docs_url}")
        return "\n".join(parts)
```

### Specific Error Types

#### UnsupportedClientError

```python
class UnsupportedClientError(MemoriError):
    """Raised when an LLM client type is not recognized."""

    def __init__(self, client_type: type, supported_types: list[type]):
        # Detect LangChain wrappers specifically
        is_langchain = client_type.__module__.startswith('langchain')

        if is_langchain:
            suggestions = [
                "Memori captures at the LLM client level, not LangChain wrappers",
                "Register the underlying client: memori.llm.register(OpenAI())",
                "Then use the registered client for calls you want captured"
            ]
        else:
            suggestions = [
                f"Supported clients: {', '.join(t.__name__ for t in supported_types)}",
                "For custom clients, implement an adapter",
            ]

        super().__init__(
            message=f"Unsupported LLM client type: {client_type.__module__}.{client_type.__name__}",
            docs_url="https://memorilabs.ai/docs/integrations",
            suggestions=suggestions
        )
```

#### DatabaseConnectionError

```python
class DatabaseConnectionError(MemoriError):
    """Raised when database connection fails."""

    def __init__(self, connection_string: str, original_error: Exception):
        # Mask password for safe display
        safe_conn = re.sub(r':([^@]+)@', ':****@', connection_string)

        super().__init__(
            message=f"Failed to connect to database: {safe_conn}",
            docs_url="https://memorilabs.ai/docs/database-setup",
            suggestions=[
                "Verify the database server is running",
                "Check connection string format matches your database type",
                "For Docker: use service name inside containers, localhost outside"
            ],
            diagnostic_steps=[
                f"Original error: {original_error}",
                "Run: python -m memori health",
                "Check firewall/network access to database host"
            ]
        )
```

#### SchemaNotInitializedError

```python
class SchemaNotInitializedError(MemoriError):
    """Raised when database schema hasn't been created."""

    def __init__(self):
        super().__init__(
            message="Database schema not initialized",
            docs_url="https://memorilabs.ai/docs/database-setup",
            suggestions=[
                "Run once: memori.config.storage.build()",
                "Or via CLI: python -m memori db init"
            ]
        )
```

### Example Output

**Before (current):**
```
RuntimeError: Unsupported LLM client type: langchain_openai.chat_models.base.ChatOpenAI
```

**After (proposed):**
```
UnsupportedClientError: Unsupported LLM client type: langchain_openai.chat_models.base.ChatOpenAI

Suggestions:
  1. Memori captures at the LLM client level, not LangChain wrappers
  2. Register the underlying client: memori.llm.register(OpenAI())
  3. Then use the registered client for calls you want captured

Documentation: https://memorilabs.ai/docs/integrations
```

## Trade-off Analysis

### Benefits

| Benefit | Impact |
|---------|--------|
| Faster resolution | Developers fix issues without searching GitHub |
| Better first experience | Errors guide instead of blocking |
| Reduced support load | Fewer duplicate issues |
| Documentation discovery | Errors link directly to relevant docs |
| Precise testing | Specific exceptions enable targeted tests |

### Costs

| Cost | Mitigation |
|------|------------|
| More code to maintain | Each exception is small (20-30 lines) |
| Docs URLs must stay in sync | Add CI check that docs URLs resolve |
| Larger exception classes | Minimal overhead; errors are infrequent |

### Backward Compatibility

Maintain backward compatibility by inheriting from existing types:

```python
class UnsupportedClientError(MemoriError, RuntimeError):
    """Inherits from RuntimeError for backward compatibility."""
    pass
```

Existing code that catches `RuntimeError` will continue to work:

```python
try:
    memori.llm.register(langchain_client)
except RuntimeError as e:  # Still catches UnsupportedClientError
    handle_error(e)
```

## Alternatives Considered

### Result Types (Rust-style)

```python
Result = Union[Success[T], Failure]
```

**Rejected:** Python ecosystem prefers exceptions. Result types add cognitive overhead.

### Error Codes with Lookup

```python
raise MemoriError(code="E001", context={...})
```

**Rejected:** Adds friction (user must look up code). Less discoverable than inline suggestions.

### Verbose Logging Instead

**Rejected:** Logs are often hidden. Exception message is still unhelpful.

## Implementation Plan

### Phase 1: Core Exceptions

1. Create `memori/exceptions.py` with base `MemoriError`
2. Implement `UnsupportedClientError` (addresses Issue #234)
3. Implement `DatabaseConnectionError`
4. Update `_registry.py` to use new exceptions

### Phase 2: Additional Exceptions

1. `ExtractionError` - for fact extraction failures
2. `RecallError` - for semantic search failures
3. `SchemaNotInitializedError` - for missing tables

### Phase 3: Documentation

1. Update docs with troubleshooting section
2. Add CI check for docs URL validity
3. Release as minor version (backward compatible)

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Time to resolve common issues | Days | Minutes |
| GitHub issues for known problems | ~5/month | ~1/month |
| Docs visits from error messages | 0 | Measurable |

## Conclusion

Structured error types represent a high-impact, medium-effort improvement. By embedding solutions directly in error messages, we reduce friction and build trust in the SDK's usability.

The trade-offs are manageable and outweighed by the benefits to developer experience.

## References

- Issue #234: LangChain RuntimeError
- Issue #238: Auto Data Capture Bug
- [Python Exception Hierarchy Best Practices](https://docs.python.org/3/library/exceptions.html)
