[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/banner-dark-large.jpg)](https://memorilabs.ai/)


<p align="center">
  <strong>Memory from what agents do, not just what they say.</strong>
</p>

<p align="center">
  <i>Memori plugs into the software and infrastructure you already use. It is LLM, datastore and framework agnostic and seamlessly integrates into the architecture you've already designed.</i>
</p>

<p align="center">
  <strong>→ <a href="https://memorilabs.ai/docs/memori-cloud/">Memori Cloud</a></strong> — Zero config. Get an API key and start building in minutes.
</p>
<p align="center">
  <a href="https://trendshift.io/repositories/15418">
    <img src="https://trendshift.io/_next/image?url=https%3A%2F%2Ftrendshift.io%2Fapi%2Fbadge%2Frepositories%2F15418&w=640&q=75" alt="Memori%2fLabs%2FMemori | Trendshif">
  </a>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/memori">
    <img src="https://badge.fury.io/py/memori.svg" alt="PyPI version">
  </a>
  <a href="https://www.npmjs.com/package/@memorilabs/memori">
    <img src="https://img.shields.io/npm/v/@memorilabs/memori.svg" alt="NPM version">
  </a>
  <a href="https://pepy.tech/projects/memori">
    <img src="https://static.pepy.tech/badge/memori" alt="Downloads">
  </a>
  <a href="https://opensource.org/license/apache-2-0">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  </a>
  <a href="https://discord.gg/abD4eGym6v">
    <img src="https://img.shields.io/discord/1042405378304004156?logo=discord" alt="Discord">
  </a>
</p>

<p align="center">
  <a href="https://github.com/MemoriLabs/Memori/stargazers">
    <img src="https://img.shields.io/badge/⭐%20Give%20a%20Star-Support%20the%20project-orange?style=for-the-badge" alt="Give a Star">
  </a>
</p>

<p align="center">
  <strong>Choose memory that performs</strong>
</p>



[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/stats.jpg)](https://memorilabs.ai/benchmark)

---

## Getting Started

### Installation

<details open>
<summary><b>TypeScript SDK</b></summary>

```bash
npm install @memorilabs/memori
```
</details>

<details>
<summary><b>Python SDK</b></summary>

```bash
pip install memori
```
</details>

### Quickstart

Sign up at [app.memorilabs.ai](https://app.memorilabs.ai), get a Memori API key, and start building. Full docs: [memorilabs.ai/docs/memori-cloud/](https://memorilabs.ai/docs/memori-cloud/).

Set `MEMORI_API_KEY` and your LLM API key (e.g. `OPENAI_API_KEY`), then:

<details open>
<summary><b>TypeScript SDK</b></summary>

```typescript
import { OpenAI } from 'openai';
import { Memori } from '@memorilabs/memori';

// Requires MEMORI_API_KEY and OPENAI_API_KEY in your environment
const client = new OpenAI();
const mem = new Memori().llm
  .register(client)
  .attribution('user_123', 'support_agent');

async function main() {
  await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'My favorite color is blue.' }],
  });
  // Conversations are persisted and recalled automatically in the background.

  const response = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: "What's my favorite color?" }],
  });
  // Memori recalls that your favorite color is blue.
}
```
</details>

<details>
<summary><b>Python SDK</b></summary>

```python
from memori import Memori
from openai import OpenAI

# Requires MEMORI_API_KEY and OPENAI_API_KEY in your environment
client = OpenAI()
mem = Memori().llm.register(client)

mem.attribution(entity_id="user_123", process_id="support_agent")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "My favorite color is blue."}]
)
# Conversations are persisted and recalled automatically.

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "What's my favorite color?"}]
)
# Memori recalls that your favorite color is blue.
```
</details>

## Explore the Memories

Use the [Dashboard](https://app.memorilabs.ai) — Memories, Analytics, Playground, and API Keys.

> [!TIP]
> Want to use your own database? Check out docs for Memori BYODB here:
> [https://memorilabs.ai/docs/memori-byodb/](https://memorilabs.ai/docs/memori-byodb/).

## LoCoMo Benchmark

Memori was evaluated on the LoCoMo benchmark for long-conversation memory and achieved **81.95% overall accuracy** while using an average of **1,294 tokens per query**. That is just **4.97% of the full-context footprint**, showing that structured memory can preserve reasoning quality without forcing large prompts into every request.

Compared with other retrieval-based memory systems, Memori outperformed Zep, LangMem, and Mem0 while reducing prompt size by roughly **67% vs. Zep** and lowering context cost by more than **20x vs. full-context prompting**.

Read the [benchmark overview](docs/memori-cloud/benchmark/overview.mdx), see the [results](docs/memori-cloud/benchmark/results.mdx), or download the [paper](https://arxiv.org/abs/2603.19935).

!["Memori's average accuracy along with the standard deviation"](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/docs/memori-locomo-benchmark.webp)

## OpenClaw (Persistent Memory for Your Gateway)

By default, OpenClaw agents forget everything between sessions. The Memori plugin fixes that. It captures durable facts and preferences after each conversation, then injects the most relevant context back into future prompts automatically.

No changes to your agent code or prompts are required. The plugin hooks into OpenClaw's lifecycle, so you get structured memory, Intelligent Recall, and Advanced Augmentation with a drop-in plugin.

```bash
openclaw plugins install @memorilabs/openclaw-memori
openclaw plugins enable openclaw-memori

openclaw config set plugins.entries.openclaw-memori.config.apiKey "YOUR_MEMORI_API_KEY"
openclaw config set plugins.entries.openclaw-memori.config.entityId "your-app-user-id"

openclaw gateway restart
```

For setup and configuration, see the [OpenClaw Quickstart](docs/memori-cloud/openclaw/quickstart.mdx). For architecture and lifecycle details, see the [OpenClaw Overview](docs/memori-cloud/openclaw/overview.mdx).

## MCP (Connect Your Agent in One Command)

Your agent forgets everything between sessions. Memori fixes that. It remembers your stack, your conventions, and how you like things done so you stop repeating yourself.

Works for solo developers and teams. Your agent learns coding patterns, reviewer preferences, and project conventions over time. For teams, that means shared context that new engineers pick up on day one instead of absorbing tribal knowledge over months.

If you use Claude Code, Cursor, Codex, Warp, or Antigravity, you can connect Memori with no SDK integration needed:

```bash
claude mcp add --transport http memori https://api.memorilabs.ai/mcp/ \
  --header "X-Memori-API-Key: ${MEMORI_API_KEY}" \
  --header "X-Memori-Entity-Id: your_username" \
  --header "X-Memori-Process-Id: claude-code"
```

For Cursor, Codex, Warp, and other clients, see the [MCP client setup guide](docs/memori-cloud/mcp/client-setup.mdx).

## Attribution

To get the most out of Memori, you want to attribute your LLM interactions to an entity (think person, place or thing; like a user) and a process (think your agent, LLM interaction or program).

If you do not provide any attribution, Memori cannot make memories for you.

<details open>
<summary><b>TypeScript SDK</b></summary>

```typescript
mem.attribution("12345", "my-ai-bot");
```
</details>

<details>
<summary><b>Python SDK</b></summary>

```python
mem.attribution(entity_id="12345", process_id="my-ai-bot")
```
</details>

## Session Management

Memori uses sessions to group your LLM interactions together. For example, if you have an agent that executes multiple steps you want those to be recorded in a single session.

By default, Memori handles setting the session for you but you can start a new session or override the session by executing the following:

<details open>
<summary><b>TypeScript SDK</b></summary>

```typescript
mem.resetSession();
// or
mem.setSession(sessionId);
```
</details>

<details>
<summary><b>Python SDK</b></summary>

```python
mem.new_session()
# or
mem.set_session(session_id)
```
</details>

## Supported LLMs

- Anthropic
- Bedrock
- DeepSeek
- Gemini
- Grok (xAI)
- OpenAI (Chat Completions & Responses API)

_(unstreamed, streamed, synchronous and asynchronous)_

## Supported Frameworks

- Agno
- LangChain
- Pydantic AI

## Supported Platforms

- DeepSeek
- Nebius AI Studio

## Examples

For more examples and demos, check out the [Memori Cookbook](https://github.com/MemoriLabs/memori-cookbook).

## Memori Advanced Augmentation

Memories are tracked at several different levels:

- **entity**: think person, place, or thing; like a user
- **process**: think your agent, LLM interaction or program
- **session**: the current interactions between the entity, process and the LLM

[Memori's Advanced Augmentation](docs/memori-cloud/concepts/advanced-augmentation.mdx) enhances memories at each of these levels with:

- attributes
- events
- facts
- people
- preferences
- relationships
- rules
- skills

Memori knows who your user is, what tasks your agent handles and creates unparalleled context between the two. Augmentation occurs in the background incurring no latency.

By default, Memori Advanced Augmentation is available without an account but rate limited. When you need increased limits, [sign up for Memori Advanced Augmentation](https://app.memorilabs.ai/signup) or use the Memori CLI:

```bash
# Install the CLI via pip to manage your account
python -m memori sign-up <email_address>
```

Memori Advanced Augmentation is always free for developers!

Once you've obtained an API key, set the following environment variable (used by both Python and TypeScript SDKs):

```bash
export MEMORI_API_KEY=[api_key]
```

## Managing Your Quota

At any time, you can check your quota using the Memori CLI (works for both SDKs):

```bash
python -m memori quota
```

Or by checking your account at [https://app.memorilabs.ai/](https://app.memorilabs.ai/). If you have reached your IP address quota, sign up and get an API key for increased limits.

If your API key exceeds its quota limits we will email you and let you know.

## Command Line Interface (CLI)

The Memori CLI is the unified tool for managing your account, keys, and quotas across all SDKs. To use it, execute the following from the command line:

```bash
# Requires Python installed
python -m memori
```

This will display a menu of the available options. For more information about what you can do with the Memori CLI, please reference [Command Line Interface](docs/memori-byodb/concepts/cli-quickstart.mdx).

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](https://github.com/MemoriLabs/Memori/blob/main/CONTRIBUTING.md) for details on:

- Setting up your development environment
- Code style and standards
- Submitting pull requests
- Reporting issues

---

## Support

- [**Memori Cloud Documentation**](https://memorilabs.ai/docs/memori-cloud)
- [**Memori BYODB Documentation**](https://memorilabs.ai/docs/memori-byodb)
- [**Discord**](https://discord.gg/FpytKAxnFb)
- [**Issues**](https://github.com/MemoriLabs/Memori/issues)
---

## License

Apache 2.0 - see [LICENSE](https://github.com/MemoriLabs/Memori/blob/main/LICENSE)
