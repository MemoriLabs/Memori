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

Install the Memori SDK and your preferred LLM client using your package manager of choice:

```bash
npm install @memorilabs/memori
```

*(Note: Memori currently supports `openai` and `@anthropic-ai/sdk` as peer dependencies).*

## Quickstart Example

```typescript
import 'dotenv/config';
import { OpenAI } from 'openai';
import { Memori } from '@memorilabs/memori';

// Environment check
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY must be set in .env');
  process.exit(1);
}

// 1. Initialize the LLM Client
const client = new OpenAI({ apiKey: OPENAI_API_KEY });

// 2. Initialize Memori and Register the Client
const memori = new Memori().llm
  .register(client)
  .attribution('typescript-sdk-test-user', 'test-process-1');

async function main() {
  console.log('--- Step 1: Teaching the AI ---');
  const factPrompt = 'My favorite color is blue and I live in Paris.';
  console.log(`User: ${factPrompt}`);

  // This call automatically triggers Persistence and Augmentation in the background.
  const response1 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: factPrompt }],
  });

  console.log(`AI:   ${response1.choices[0].message.content}`);

  console.log('\n(Waiting 5 seconds for backend processing...)\n');
  await new Promise((resolve) => setTimeout(resolve, 5000));

  console.log('--- Step 2: Testing Recall ---');
  const questionPrompt = 'What is my favorite color?';
  console.log(`User: ${questionPrompt}`);

  // This call automatically triggers Recall, injecting the Paris/Blue facts into the prompt.
  const response2 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: questionPrompt }],
  });

  console.log(`AI:   ${response2.choices[0].message.content}`);
}

main().catch(console.error);
```

## Key Features

- **Zero-Latency Memory:** Background processing ensures your LLM calls are never slowed down.
- **Advanced Augmentation:** Automatically extracts and structures facts, preferences, and relationships.
- **Cloud-Hosted:** Fully managed infrastructure via the Memori Cloud API.
- **LLM Agnostic:** Native support for the official OpenAI and Anthropic SDKs via interceptors.
- **Automatic Prompt Injection:** Seamlessly fetches relevant memories and injects them into the system context.

## Attribution

To get the most out of Memori, you want to attribute your LLM interactions to an entity (think person, place or thing; like a user) and a process (think your agent, LLM interaction or program).

If you do not provide any attribution, Memori cannot make memories for you.

```typescript
memori.attribution('user-12345', 'my-ai-bot');
```

## Session Management

Memori uses sessions to group your LLM interactions together. For example, if you have an agent that executes multiple steps you want those to be recorded in a single session.

By default, Memori handles setting the session for you but you can start a new session or override the session by executing the following:

```typescript
memori.resetSession();
```

or

```typescript
const sessionId = memori.session.id;

// ... Later ...

memori.setSession(sessionId);
```

## Supported LLMs

- Anthropic Claude (`@anthropic-ai/sdk`)
- OpenAI (`openai`)

## Memori Advanced Augmentation

Memories are tracked at several different levels:

- **entity**: think person, place, or thing; like a user
- **process**: think your agent, LLM interaction or program
- **session**: the current interactions between the entity, process and the LLM

[Memori's Advanced Augmentation](https://github.com/MemoriLabs/Memori/blob/main/docs/advanced-augmentation.md) enhances memories at each of these levels with:

- attributes
- events
- facts
- people
- preferences
- relationships
- rules
- skills

Memori knows who your user is, what tasks your agent handles and creates unparalleled context between the two. Augmentation occurs asynchronously in the background incurring no latency.

By default, Memori Advanced Augmentation is available without an account but is rate limited. When you need increased limits, [sign up for Memori Advanced Augmentation](https://app.memorilabs.ai/signup).

Memori Advanced Augmentation is always free for developers!

Once you've obtained an API key, simply set the following environment variable:

```bash
export MEMORI_API_KEY=[api_key]
```

## Managing Your Quota

You can check your quota and manage your account by logging in at [https://memorilabs.ai/](https://memorilabs.ai/). If you have reached your IP address quota, sign up and get an API key for increased limits.

If your API key exceeds its quota limits we will email you and let you know.

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