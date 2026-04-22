/**
 * Memori + MikroORM + SQLite
 *
 * Same narrative as examples/sqlite/main.ts; shows passing a MikroORM `EntityManager`.
 */

import 'dotenv/config';
import { MikroORM, EntitySchema } from '@mikro-orm/core';
import { SqliteDriver } from '@mikro-orm/sqlite';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

// MikroORM strictly requires at least one entity to boot.
// Since Memori manages its own raw tables, we create a dummy schema
// just to satisfy the MikroORM initialization in this isolated test.
const DummyEntity = new EntitySchema({
  name: 'Dummy',
  properties: {
    id: { type: 'number', primary: true },
  },
});

async function main(): Promise<void> {
  const orm = await MikroORM.init({
    driver: SqliteDriver,
    dbName: 'memori-mikro.db',
    entities: [DummyEntity],
    allowGlobalContext: true,
  });

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  const mem = new Memori({ conn: orm.em }).llm.register(client);
  mem.attribution('user-123', 'my-app');

  if (!mem.config.storage) {
    throw new Error('Storage not initialized');
  }

  try {
    await mem.config.storage.build();

    console.log('You: My favorite color is blue and I live in Paris');
    const response1 = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'My favorite color is blue and I live in Paris' }],
    });
    console.log(`AI: ${response1.choices[0]?.message?.content}\n`);

    await mem.engine.waitForAugmentation();

    console.log("You: What's my favorite color?");
    const response2 = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: "What's my favorite color?" }],
    });
    console.log(`AI: ${response2.choices[0]?.message?.content}\n`);

    console.log('You: What city do I live in?');
    const response3 = await client.chat.completions.create({
      model: 'gpt-4o-mini',
      messages: [{ role: 'user', content: 'What city do I live in?' }],
    });
    console.log(`AI: ${response3.choices[0]?.message?.content}`);

    await mem.engine.waitForAugmentation();
  } finally {
    await mem.config.storage.close();
    await orm.close();
  }
}

main().catch((err: unknown) => {
  console.error(err);
  process.exitCode = 1;
});
