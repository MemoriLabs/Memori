# Memori for Hermes Agent

Memori gives Hermes Agent structured long-term memory. It recalls relevant context before each turn, captures completed user/assistant exchanges after each turn, and exposes explicit tools for memory search, summaries, quota checks, signup, and feedback.

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
pip install memori-hermes
```

## Configure

Use Hermes' memory setup flow and select `memori`:

```bash
hermes memory setup
```

If `memori` is not listed yet, install `memori-hermes` in the same Python environment Hermes uses, then set the provider manually.

Manual configuration also works:

```bash
hermes config set memory.provider memori
echo "MEMORI_API_KEY=your-key" >> ~/.hermes/.env
```

Then add `~/.hermes/memori.json`:

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
- `MEMORI_RECALL_LIMIT`

## Tools

- `memori_recall`
- `memori_recall_summary`
- `memori_quota`
- `memori_signup`
- `memori_feedback`

## Behavior

The provider is intentionally fail-soft. Memori network failures are logged but do not stop Hermes from answering the user.
