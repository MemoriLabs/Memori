[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/banner.png)](https://memorilabs.ai/)

<p align="center">
  <strong>The memory fabric for enterprise AI</strong>
</p>

<p align="center">
  <i>Memori plugs into the software and infrastructure you already use. It is LLM and framework agnostic and seamlessly integrates into the architecture you've already designed.</i>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@memorilabs/memori">
    <img src="https://img.shields.io/npm/v/@memorilabs/memori.svg" alt="NPM version">
  </a>
  <a href="https://www.npmjs.com/package/@memorilabs/memori">
    <img src="https://img.shields.io/npm/dm/@memorilabs/memori.svg" alt="NPM Downloads">
  </a>
  <a href="https://opensource.org/license/apache-2-0">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  </a>
  <a href="https://discord.gg/abD4eGym6v">
    <img src="https://img.shields.io/discord/1042405378304004156?logo=discord" alt="Discord">
  </a>
</p>

---

## Getting Started

Install the Memori TypeScript SDK using your preferred package manager:

```bash
npm install @memorilabs/memori
# or
yarn add @memorilabs/memori
# or
pnpm add @memorilabs/memori
```

## Quickstart Example

Memori works effortlessly with your existing LLM clients. Just register your client, and Memori handles the context injection, persistence, and augmentation automatically in the background.

```typescript
import { Memori } from '@memorilabs/memori';
import OpenAI from 'openai';

// 1. Initialize your LLM client
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// 2. Initialize Memori and register the client
const memori = new Memori();
memori.llm.register(client);

// 3. Set the attribution (who the user is, and what the process is)
memori.attribution('user_123456', 'test-ai-agent');

async function run() {
  // Memori intercepts the call, persists the message, and extracts memories
  const response1 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'My favorite color is blue.' }],
  });
  console.log(response1.choices[0].message.content);

  // In subsequent calls, Memori automatically recalls relevant context
  // and injects it into the system prompt!
  const response2 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: "What's my favorite color?" }],
  });
  console.log(response2.choices[0].message.content);
}

run();
```

## Attribution

To get the most out of Memori, you need to attribute your LLM interactions to an **entity** (think person, place, or thing; like a user) and a **process** (think your agent, LLM interaction, or program).

If you do not provide any attribution, Memori cannot properly segregate or retrieve memories for you.

```typescript
memori.attribution('user_12345', 'my-ai-bot');
```

## Session Management

Memori uses sessions to group your LLM interactions together. For example, if you have an agent that executes multiple steps, you want those to be recorded in a single session.

By default, Memori handles setting a session UUID for you, but you can easily reset it or override it (e.g., when resuming a chat from a database):

```typescript
// Start a brand new session with a fresh UUID
memori.resetSession();

// ... or resume an existing session
memori.setSession('your-existing-uuid-here');
```

## Manual Recall

If you need to fetch memories explicitly without triggering a full LLM completion, you can use the manual recall method:

```typescript
const facts = await memori.recall("What is the user's favorite color?");
console.log(facts);
// Returns a list of ParsedFact objects with content, relevance scores, and timestamps
```

## Supported Providers

The TypeScript SDK currently supports the following LLM providers out-of-the-box via `@memorilabs/axon`:

- **OpenAI** (Chat Completions & Legacy Responses API)
- **Anthropic** (Messages API)

## Memori Advanced Augmentation (Hosted Cloud)

The TypeScript SDK connects to the Memori Cloud to handle advanced memory augmentation. Memories are tracked and enhanced at several different levels (entities, processes, and sessions) to extract:

- attributes
- events
- facts
- people
- preferences
- relationships
- rules
- skills

Memori knows who your user is, what tasks your agent handles, and creates unparalleled context between the two. Augmentation occurs asynchronously in the background, incurring no added latency to your LLM calls.

### API Keys and Quotas

By default, Memori Advanced Augmentation is available without an account but is rate-limited. Memori Advanced Augmentation is **always free for developers!**

To increase your limits, [sign up for a Memori API Key](https://app.memorilabs.ai/signup) and set it in your environment:

```bash
export MEMORI_API_KEY=your_api_key_here
```

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](https://github.com/MemoriLabs/Memori/blob/main/CONTRIBUTING.md) for details on:

- Setting up your development environment
- Code style and standards
- Submitting pull requests
- Reporting issues

---

## Support

- **Documentation**: [https://memorilabs.ai/docs](https://memorilabs.ai/docs)
- **Discord**: [https://discord.gg/abD4eGym6v](https://discord.gg/abD4eGym6v)
- **Issues**: [GitHub Issues](https://github.com/MemoriLabs/Memori/issues)

---

## License

Apache 2.0 - see [LICENSE](https://github.com/MemoriLabs/Memori/blob/main/LICENSE)
