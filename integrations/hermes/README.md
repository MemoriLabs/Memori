# Memori for Hermes Agent

Official Memori Labs provider for Hermes Agent, enabling structured long-term memory from agent trace and execution.

Memori gives Hermes Agent structured, long-term memory from agent trace and execution. It captures completed agent activity — including user goals, assistant decisions, tool usage, workflow steps, outcomes, constraints, failures, and feedback — and structures that activity into durable memory primitives for future recall. This allows Hermes agents to learn from prior execution, preserve workflow context, avoid repeated mistakes, and become more efficient over time. The provider exposes explicit tools for memory recall, summaries, quota checks, signup, and feedback.

## Requirements

- Hermes Agent with memory provider plugins
- A Memori API key
- Python 3.10+

## Install

From this repository:

```bash
pip install -e .
pip install -e integrations/hermes
```

Or install the published package when available:

```bash
pip install hermes-memori
```

## Configure

Use Hermes' memory setup flow and select `memori`:

```bash
hermes memory setup
```

If `memori` is not listed yet, install `hermes-memori` in the same Python environment Hermes uses, then set the provider manually.

Manual configuration also works:

```bash
hermes config set memory.provider memori
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
mkdir -p "$HERMES_HOME"
echo "MEMORI_API_KEY=your-key" >> "$HERMES_HOME/.env"
```

Then add `$HERMES_HOME/memori.json`:

```json
{
  "entityId": "your-user-or-workspace-id",
  "projectId": "hermes"
}
```

Environment variables override file config:

- `MEMORI_API_KEY`
- `MEMORI_ENTITY_ID`
- `MEMORI_PROJECT_ID`
- `MEMORI_PROCESS_ID`

`MEMORI_PROJECT_ID` is optional. When it is not configured, Hermes-provided project context such as the active workspace, agent identity, user ID, session title, or session ID is used as the Memori project scope.

## Tools

- `memori_recall`
- `memori_recall_summary`
- `memori_quota`
- `memori_signup`
- `memori_feedback`

## Behavior

The provider is intentionally fail-soft. Memori network failures are logged but do not stop Hermes from answering the user.
