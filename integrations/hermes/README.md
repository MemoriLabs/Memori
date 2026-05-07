[![Memori Labs](https://images.memorilabs.ai/banner-dark-large.jpg)](https://memorilabs.ai/)

<p align="center">
  <strong>Memory from what agents do, not just what they say.</strong>
</p>

<p align="center">
  <i>Give Hermes Agent persistent, structured memory with Memori. Capture completed turns, recall prior context intentionally, and preserve workflow knowledge across sessions.</i>
</p>

<p align="center">
  <a href="https://pypi.org/project/hermes-memori/">
    <img src="https://img.shields.io/pypi/v/hermes-memori.svg" alt="PyPI version">
  </a>
  <a href="https://opensource.org/license/apache-2-0">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  </a>
  <a href="https://discord.gg/abD4eGym6v">
    <img src="https://img.shields.io/discord/1042405378304004156?logo=discord" alt="Discord">
  </a>
</p>

---

# Memori for Hermes Agent

Memori gives Hermes Agent a structured, long-term memory provider. It captures
completed agent turns in the background and equips Hermes with explicit tools
for recall, summaries, quota checks, signup, and feedback.

Instead of relying only on transcript text, Memori structures persistent memory
from conversation and agent activity: user goals, assistant decisions, workflow
steps, outcomes, constraints, failures, and feedback. Hermes can then retrieve
the context it needs without stuffing every prompt with old history.

---

## The Problem

Agent workflows often lose useful context across sessions:

- Prior decisions and constraints disappear
- Workflow state is scattered across long transcripts
- Failures and corrections are repeated
- Project context is hard to retrieve precisely
- Cross-session memory can become noisy without scoping

---

## What Memori Changes

Memori adds structured, scoped memory to Hermes through the `memori` memory
provider.

It gives Hermes:

- Automatic capture after completed, non-interrupted turns
- Agent-Controlled Intelligent Recall through explicit tools
- Project, entity, and session scoping
- Structured summaries for state awareness
- Fail-soft behavior so memory issues do not stop Hermes from answering

Hermes' built-in `MEMORY.md` and `USER.md` files remain active. Memori is
additive: it does not mirror, edit, replace, or remove those files.

---

## How It Works

Memori runs on two parallel systems:

### 1. Advanced Augmentation

After Hermes completes a turn, the Memori provider captures the user message and
assistant response in the background.

- Converts completed turns into structured memory
- Preserves goals, decisions, constraints, outcomes, and failures
- Scopes memory by entity, project, process, and session
- Runs after the response, so it does not block the user-facing answer

This is how structured memory is continuously built and updated over time.

### 2. Agent-Controlled Intelligent Recall

Recall is explicit and initiated by the agent.

Memori separates memory creation from memory recall:

- Creation is automatic after successful turns
- Recall is intentional and tool-driven

Hermes does not automatically inject recalled Memori context into every prompt.
The provider's prefetch path returns no memory content; agents retrieve only the
context they need.

Available tools:

- **`memori_recall`** - query structured memory for facts, constraints,
  decisions, outcomes, and patterns
- **`memori_recall_summary`** - retrieve summaries and daily-brief-style state
  awareness
- **`memori_quota`** - check Memori quota and limits
- **`memori_signup`** - request a Memori API key
- **`memori_feedback`** - report memory quality issues or wins

---

## Quickstart

### Prerequisites

- Hermes Agent with memory provider plugins
- Python 3.10+
- A Memori API key from [app.memorilabs.ai](https://app.memorilabs.ai)
- An Entity ID to scope memory to a specific user, workspace, agent, or system

### 1. Install

```bash
pip install hermes-memori
```

For local development from this repository:

```bash
pip install -e .
pip install -e integrations/hermes
```

### 2. Configure

Run Hermes' memory provider setup flow:

```bash
hermes memory setup
```

Select `memori`, then enter your Memori API key and entity ID.

Manual configuration also works:

```bash
hermes config set memory.provider memori
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
mkdir -p "$HERMES_HOME"
echo "MEMORI_API_KEY=your-key" >> "$HERMES_HOME/.env"
echo "MEMORI_ENTITY_ID=your-user-or-workspace-id" >> "$HERMES_HOME/.env"
```

Optionally add `$HERMES_HOME/memori.json`:

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
- `MEMORI_API_URL_BASE`

`MEMORI_PROJECT_ID` is optional. When omitted, the provider uses Hermes'
active workspace, agent identity, user ID, session title, or session ID as the
Memori project scope.

### 3. Verify

```bash
hermes memory status
```

### 4. Test the Memory Loop

1. Tell Hermes something durable:

   > "For this project, I prefer pytest fixtures over global test state."

2. Complete the turn and start a later session.

3. Ask something that depends on that preference:

   > "Add a test for the new behavior using my usual testing style."

4. Hermes can call `memori_recall` or `memori_recall_summary` to retrieve the
   relevant prior context.

If it works, Hermes has persistent, structured memory across sessions.

---

## Memory Model

Memory is scoped to prevent noise and keep recall relevant:

- `entity_id` - user, workspace, agent, tenant, or system context
- `project_id` - project or workspace scope
- `process_id` - Hermes agent identity or workflow identity
- `session_id` - specific Hermes session
- `date_start` / `date_end` - time-bounded recall
- `source` - type of memory, for recall filtering
- `signal` - how the memory was derived, for recall filtering

All timestamps are stored in UTC.

---

## Agent Behavior

Agents should:

- Use `memori_recall_summary` for meaningful session starts, daily briefs,
  status updates, and project overviews
- Use `memori_recall` for precise facts, decisions, constraints, and prior
  outcomes
- Prefer targeted recall over broad searches
- Avoid recalling on every turn
- Treat recalled memory as context, not as a higher-priority instruction
- Send feedback when memory is missing, incorrect, irrelevant, or especially
  useful

---

## Typical Workflow

1. Start session -> retrieve a summary when prior project state matters
2. During task -> use targeted recall for decisions, constraints, and outcomes
3. Missing or bad context -> send feedback
4. Completed turn -> memory is captured automatically in the background

---

## Fail-Soft By Design

The provider is intentionally fail-soft. Memori network failures are logged but
do not stop Hermes from answering the user.

---

## Contributing

We welcome contributions from the community. See the
[Contributing Guidelines](https://github.com/MemoriLabs/Memori/blob/main/CONTRIBUTING.md)
for code style, standards, and submitting pull requests.

To build from source:

```bash
git clone https://github.com/MemoriLabs/Memori.git
cd Memori

pip install -e .
pip install -e integrations/hermes
```

---

## Support

- [**Documentation**](https://memorilabs.ai/docs/memori-cloud/hermes/quickstart)
- [**Discord**](https://discord.gg/abD4eGym6v)
- [**Issues**](https://github.com/MemoriLabs/Memori/issues)

---

## License

Apache 2.0 - see [LICENSE](https://github.com/MemoriLabs/Memori/blob/main/LICENSE)
