[![Memori Labs](https://images.memorilabs.ai/banner-dark-large.jpg)](https://memorilabs.ai/)

<p align="center">
  <strong>Memory from what agents do, not just what they say.</strong>
</p>

<p align="center">
  <i>Give Hermes Agent persistent, structured memory with Memori. Capture agent trace, tool activity, decisions, outcomes, and conversation into durable memory that Hermes can recall across sessions.</i>
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

Memori gives Hermes Agent a structured, long-term memory provider that captures not only conversation, but also agent trace and execution context. As Hermes completes work, Memori can structure durable memory from the agent’s actions: tool calls, workflow steps, assistant decisions, outcomes, failures, constraints, and feedback.

This means Hermes can remember what happened during prior execution paths, not just what was said in the transcript. Instead of stuffing old conversation history into every prompt, Hermes can intentionally retrieve the structured context it needs to continue work, avoid repeated mistakes, preserve project knowledge, and improve across sessions.

---

## The Problem

Agent workflows often lose useful context across sessions:

- Prior decisions and constraints disappear
- Workflow state is scattered across long transcripts
- Conversation-only memory misses tool calls, workflow decisions, outcomes, failures, and corrections
- Failures and corrections are repeated
- Project context is hard to retrieve precisely
- Cross-session memory can become noisy without scoping

---

## What Memori Changes

Memori adds structured, scoped, agent-native memory to Hermes through the `memori` memory provider.

It gives Hermes:

- Automatic capture after completed, non-interrupted turns
- Structured memory from agent trace, execution paths, decisions, outcomes, and conversation
- Agent-Controlled Intelligent Recall through explicit tools
- Project, entity, process, and session scoping
- Structured summaries for state awareness
- Fail-soft behavior so memory issues do not stop Hermes from answering

Hermes' built-in `MEMORY.md` and `USER.md` files remain active. Memori is additive: it does not mirror, edit, replace, or remove those files.

---

## How It Works

Memori runs on two parallel systems:

### 1. Advanced Augmentation

After Hermes completes a turn, the Memori provider captures the user message, assistant response, and available execution context in the background.

Memori then converts the completed interaction into structured memory primitives, including:

- User goals and preferences
- Assistant decisions and reasoning outcomes
- Tool calls and execution steps
- Workflow state and task progress
- Constraints, instructions, and project-specific rules
- Results, failures, corrections, and recurring patterns

Memory is scoped by entity, project, process, and session, then updated asynchronously after the response so it does not block the user-facing answer.

This is how Hermes continuously builds memory from what it says and what it does.

### 2. Agent-Controlled Intelligent Recall

Memori separates memory creation from memory recall:

- **Creation** is automatic (advanced augmentation).
- **Recall** is intentional (agent-controlled).

Agents decide:

- When to recall
- What scope to recall from
- How much history to include

Recall is also intelligent. When memories are retrieved, Memori does not simply return raw chronological history. It uses a proprietary multi-dimensional ranking algorithm to prioritize the memories most likely to matter for the current agent task.

The recall algorithm takes into account factors such as:

- Source and signal weights
- Recency
- Frequency
- Memory type
- Scope relevance
- Historical importance
- Retrieval context

This allows agents to retrieve the most relevant facts, decisions, constraints, patterns, and prior outcomes without stuffing irrelevant history into the prompt.

**Supported parameters:** `entity_id`, `project_id`, `session_id`, `date_start`, `date_end`, `source`, `signal`

**Memory classification schema**

*Sources*

- `constraint`
- `decision`
- `execution`
- `fact`
- `insight`
- `instruction`
- `status`
- `strategy`
- `task`

*Signals*

- `commit`
- `discovery`
- `failure`
- `inference`
- `pattern`
- `result`
- `update`
- `verification`

**Default behavior:** If no date range is provided, recall returns all-time memories.

**Returned context may include:**

- Relevant facts
- Prior decisions
- Constraints
- Patterns
- Summaries
- Execution outcomes
- Known failure modes

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
