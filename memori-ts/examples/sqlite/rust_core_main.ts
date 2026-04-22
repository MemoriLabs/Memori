/**
 * SQLite BYODB + Rust core smoke test.
 *
 * This mirrors examples/sqlite/rust_core_main.py but uses the TypeScript SDK's
 * native engine (no separate feature flag — build native bindings with `npm run sync-native`).
 */

import 'dotenv/config';
import { OpenAI } from 'openai';
import Database from 'better-sqlite3';
import { Memori } from '../../src/index.js';

async function main(): Promise<void> {
  console.log('Starting rust_core_main...');
  const openaiApiKey = process.env.OPENAI_API_KEY;
  if (!openaiApiKey) {
    throw new Error('Set OPENAI_API_KEY before running this example.');
  }

  process.env.MEMORI_TEST_MODE = '1';

  const client = new OpenAI({ apiKey: openaiApiKey, timeout: 30_000 });
  const db = new Database('memori_rust_core.db');

  console.log('Initializing Memori (BYODB + Rust core)...');
  const mem = new Memori({ conn: db }).llm.register(client);
  mem.attribution('rust-core-user', 'sqlite-example');

  if (!mem.config.storage) {
    throw new Error('Storage not initialized');
  }

  try {
    await mem.config.storage.build();

    if (!mem.engine.hasStorage) {
      throw new Error('Storage bridge is not active. Pass conn= to Memori.');
    }

    console.log('\nYou: What is my favorite season?');
    const first = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'What is my favorite season?' }],
    });
    console.log('AI:', first.choices[0]?.message?.content);

    console.log("\nYou: What's my favorite season?");
    const second = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: "What's my favorite season?" }],
    });
    console.log('AI:', second.choices[0]?.message?.content);

    console.log('\nYou: What season do I like for the weather?');
    const third = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'What season do I like for the weather?' }],
    });
    console.log('AI:', third.choices[0]?.message?.content);

    await mem.augmentation.wait(10_000);
    console.log('\nDone.');
  } finally {
    await mem.config.storage.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exitCode = 1;
});
