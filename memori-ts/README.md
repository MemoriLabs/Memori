[![Memori Labs](https://images.memorilabs.ai/banner-dark-large.jpg)](https://memorilabs.ai/)

<p align="center">
  <strong>Memory from what agents do, not just what they say.</strong>
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

<p align="center">
  <a href="https://github.com/MemoriLabs/Memori/stargazers">
    <img src="https://img.shields.io/badge/⭐%20Give%20a%20Star-Support%20the%20project-orange?style=for-the-badge" alt="Give a Star">
  </a>
</p>

<p align="center">
  <strong>Choose memory that performs</strong>
</p>

[![Memori Labs](https://images.memorilabs.ai/stats.jpg)](https://memorilabs.ai/benchmark)

---

## Getting Started

Install the Memori SDK and your preferred LLM client using your package manager of choice:

```bash
npm install @memorilabs/memori
```

_(Memori supports `openai`, `@anthropic-ai/sdk`, and `@google/genai` as peer dependencies. Requires Node.js 18 or higher.)_

## Quickstart

### Memori Cloud

Zero config. Sign up at [app.memorilabs.ai](https://app.memorilabs.ai), set `MEMORI_API_KEY` and your LLM key, then:

```typescript
import 'dotenv/config';
import { OpenAI } from 'openai';
import { Memori } from '@memorilabs/memori';

const client = new OpenAI();
const mem = new Memori().llm.register(client).attribution('user_123', 'my-agent');

await client.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'My favorite color is blue.' }],
});
// Conversations are persisted and recalled automatically.

const response = await client.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: "What's my favorite color?" }],
});
// Memori recalls that your favorite color is blue.
```

### BYODB (Bring Your Own Database)

Self-host with your own database. Install a database driver alongside Memori:

```bash
npm install @memorilabs/memori better-sqlite3
```

```typescript
import 'dotenv/config';
import Database from 'better-sqlite3';
import { OpenAI } from 'openai';
import { Memori } from '@memorilabs/memori';

const db = new Database('memori.db');
const client = new OpenAI();

const mem = new Memori({ conn: () => db }).llm.register(client);
mem.attribution('user_123', 'my-agent');

if (!mem.config.storage) {
  throw new Error('Storage not initialized');
}

// Run once on startup to create Memori's schema tables
await mem.config.storage.build();

await client.chat.completions.create({
  model: 'gpt-4o-mini',
  messages: [{ role: 'user', content: 'My favorite color is blue.' }],
});

// In short-lived scripts, wait for background augmentation before exiting
await mem.augmentation.wait();

// Close your own database connection — Memori handles engine cleanup automatically
db.close();
```

> [!TIP]
> Want the full BYODB setup guide? Check out the docs:
> [https://memorilabs.ai/docs/memori-byodb/](https://memorilabs.ai/docs/memori-byodb/)

## Key Features

- **Zero-Latency Memory:** Background processing ensures your LLM calls are never slowed down.
- **Advanced Augmentation:** Automatically extracts and structures facts, preferences, and relationships.
- **Memori Cloud:** Fully managed infrastructure via the Memori Cloud API — no database required.
- **BYODB:** Self-host with your own database. SQLite, PostgreSQL, and MySQL are all supported. Pass any ORM's underlying connection pool and it works out of the box.
- **LLM Agnostic:** Native support for OpenAI, Anthropic, and Google Gemini via interceptors.
- **Automatic Prompt Injection:** Seamlessly fetches relevant memories and injects them into the system context.

## Attribution

To get the most out of Memori, attribute your LLM interactions to an entity (think person, place, or thing — like a user) and a process (think your agent, LLM interaction, or program).

If you do not provide attribution, Memori cannot make memories for you.

```typescript
mem.attribution('user-123', 'my-app');
```

## Session Management

Memori uses sessions to group your LLM interactions together. For example, if you have an agent that executes multiple steps you want those recorded in a single session.

By default, Memori handles sessions for you, but you can start a new session or resume an existing one:

```typescript
mem.resetSession();
```

```typescript
const sessionId = mem.session.id;

// ... Later ...

mem.setSession(sessionId);
```

## Supported LLMs

- Anthropic Claude (`@anthropic-ai/sdk`)
- OpenAI (`openai`)
- Google Gemini (`@google/genai`)

## Supported Databases (BYODB)

**Raw Drivers**

| Driver           | Dialects                |
| ---------------- | ----------------------- |
| `better-sqlite3` | SQLite                  |
| `pg`             | PostgreSQL, CockroachDB |
| `mysql2`         | MySQL, MariaDB          |

**Using an ORM?** Memori needs a direct connection factory — but you're already creating a raw pool for your ORM. Pass that same pool to Memori and both share it with no conflict:

```typescript
// You already have this for Drizzle
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });
const db = drizzle(pool);

// Just also give Memori the pool — no extra connection needed
const mem = new Memori({ conn: () => pool });
```

The same pattern applies to Sequelize (`mysql.createPool(...)`), MikroORM (`new pg.Pool(...)`), and any other ORM. Your ORM handles your queries; Memori handles its own tables — same pool, no conflict.

## Memori Advanced Augmentation

Memories are tracked at several different levels:

- **entity**: think person, place, or thing; like a user
- **process**: think your agent, LLM interaction, or program
- **session**: the current interactions between the entity, process, and the LLM

[Memori's Advanced Augmentation](https://memorilabs.ai/docs/memori-byodb/concepts/advanced-augmentation) enhances memories at each of these levels with:

- attributes
- facts
- preferences
- skills

Memori knows who your user is, what tasks your agent handles, and creates unparalleled context between the two. Augmentation occurs asynchronously in the background incurring no latency.

By default, Memori Advanced Augmentation is available without an account but is rate limited. When you need increased limits, [sign up for Memori Advanced Augmentation](https://app.memorilabs.ai/signup).

Memori Advanced Augmentation is always free for developers!

Once you've obtained an API key, set the following environment variable:

```bash
export MEMORI_API_KEY=[api_key]
```

## Sign Up and Managing Your Quota

You can sign up and manage your quota using the Memori CLI:

```bash
# If installed locally to your project
npx memori sign-up your-email@example.com
npx memori quota

# If installed globally (npm install -g @memorilabs/memori)
memori sign-up your-email@example.com
memori quota
```

Or by logging in at [https://app.memorilabs.ai/](https://app.memorilabs.ai/). If you have reached your IP address quota, sign up and get an API key for increased limits.

If your API key exceeds its quota limits we will email you and let you know.

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
