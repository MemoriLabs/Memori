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

async function runMikroTest() {
  console.log('🚀 1. Initializing Memori with MikroORM (SQLite)...');

  const orm = await MikroORM.init({
    driver: SqliteDriver,
    dbName: 'memori-mikro.db',
    entities: [DummyEntity], // Inject the dummy entity here
    allowGlobalContext: true,
  });

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  // Pass the EntityManager (orm.em) to Memori
  const mem = new Memori({ conn: orm.em }).llm.register(client);
  mem.attribution('mikro-user', 'mikro-test');

  console.log('🧱 2. Building Database Schema via MikroORM Adapter...');
  if (!mem.config.storage) throw new Error('Storage not initialized');
  await mem.config.storage.build();

  console.log('\n💬 3. Sending teaching message...');
  await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'I play the acoustic guitar.' }],
  });

  console.log('\n⏳ 4. Waiting for Rust engine...');
  await mem.engine.waitForAugmentation();

  // Verification using MikroORM's raw connection
  const entities = await orm.em
    .getConnection()
    .execute('SELECT id, external_id FROM memori_entity');
  const facts = await orm.em
    .getConnection()
    .execute('SELECT id, entity_id, content FROM memori_entity_fact');

  console.log(`\n[DB Check] Entities: ${entities.length}, Facts: ${facts.length}`);

  console.log('\n🧠 5. Testing Recall...');
  const query = 'What instrument do I play?';
  const recalled = await mem.recall(query);

  if (recalled.length === 0) {
    console.log('   [Recall] No memories found.');
  } else {
    console.log(`   [Recall] ${recalled.length} fact(s) retrieved:`);
    for (const fact of recalled) {
      console.log(`     • [${fact.score.toFixed(3)}] ${fact.content}`);
    }
  }

  const response = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: query }],
  });
  console.log(`\n🤖 AI: ${response.choices[0]?.message.content}`);

  await mem.engine.waitForAugmentation();

  console.log('\n🧹 6. Cleaning up...');
  await mem.config.storage.close();
  await orm.close();
  console.log('✅ MikroORM Test Complete!');
}

runMikroTest().catch(console.error);
