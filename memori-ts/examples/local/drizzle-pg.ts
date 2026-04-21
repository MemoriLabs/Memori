import 'dotenv/config';
import pg from 'pg';
import { drizzle } from 'drizzle-orm/node-postgres';
import { sql } from 'drizzle-orm';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

async function runDrizzleTest() {
  console.log('🚀 1. Initializing Memori with Drizzle ORM (Postgres)...');

  const pool = new pg.Pool({
    host: 'localhost',
    user: 'memori',
    database: 'memori_test',
    password: 'memori',
    port: 5432,
  });

  const db = drizzle(pool);
  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  // Pass the Drizzle DB instance directly to Memori
  const mem = new Memori({ conn: db }).llm.register(client);
  mem.attribution('drizzle-user', 'drizzle-test');

  console.log('🧱 2. Building Database Schema via Drizzle Adapter...');
  if (!mem.config.storage) throw new Error('Storage not initialized');
  await mem.config.storage.build();

  console.log('\n💬 3. Sending teaching message...');
  await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'My favorite programming language is Rust.' }],
  });

  console.log('\n⏳ 4. Waiting for Rust engine...');
  await mem.engine.waitForAugmentation();

  // Verification using Drizzle's public API
  const eRes = await db.execute(sql.raw('SELECT id, external_id FROM memori_entity'));
  const fRes = await db.execute(sql.raw('SELECT id, entity_id, content FROM memori_entity_fact'));

  const entities = (eRes as { rows: unknown[] }).rows;
  const facts = (fRes as { rows: unknown[] }).rows;
  console.log(`\n[DB Check] Entities: ${entities.length}, Facts: ${facts.length}`);

  console.log('\n🧠 5. Testing Recall...');
  const query = 'What programming language do I like?';
  const recalled = await mem.recall(query);

  if (recalled.length === 0) {
    console.log('   [Recall] No memories found.');
  } else {
    console.log(`   [Recall] ${recalled.length} fact(s) retrieved:`);
    for (const fact of recalled) {
      const score = fact.score.toFixed(3);
      console.log(`     • [${score}] ${fact.content}`);
    }
  }

  const response = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: query }],
  });
  console.log(`\n🤖 AI: ${response.choices[0]?.message.content}`);

  await mem.engine.waitForAugmentation();

  console.log('\n🧹 6. Cleaning up...');
  await mem.config.storage.close(); // Closes via the adapter
  await pool.end(); // Kill the underlying pg pool
  console.log('✅ Drizzle Test Complete!');
}

runDrizzleTest().catch(console.error);
