import 'dotenv/config';
import { OpenAI } from 'openai';
import Database from 'better-sqlite3';
import { Memori } from '../../src/index.js';

// 1. Initialize standard OpenAI client
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

// 2. Create a local SQLite database file
const db = new Database('memori-test.db');

async function runLocalTest() {
  console.log('🚀 1. Initializing Memori with local SQLite...');
  const mem = new Memori({ conn: db }).llm.register(client);

  mem.attribution('test-user-001', 'local-test-script');

  console.log('🧱 2. Building Database Schema...');
  if (!mem.config.storage) throw new Error('Storage not initialized');
  await mem.config.storage.build();

  console.log('\n💬 3. Sending first message (Teaching Memori)...');
  const response1 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [
      {
        role: 'user',
        content: 'Hi! My name is Ryan and my absolute favorite food is spicy tonkotsu ramen.',
      },
    ],
  });
  console.log(`🤖 AI: ${response1.choices[0].message.content}`);

  console.log('\n⏳ 4. Waiting for Rust engine to process the memory locally...');
  await mem.engine.waitForAugmentation();

  // Direct DB check — bypasses the Rust engine entirely so we know what was written.
  const entityRows = db.prepare('SELECT id, external_id FROM memori_entity').all() as Array<{
    id: number;
    external_id: string;
  }>;
  const factRows = db
    .prepare(
      'SELECT id, entity_id, content, length(content_embedding) as emb_bytes FROM memori_entity_fact'
    )
    .all() as Array<{ id: number; entity_id: number; content: string; emb_bytes: number }>;
  console.log(`\n[DB Check] Entities stored: ${entityRows.length}`);
  for (const e of entityRows) console.log(`  - [${e.id}] ${e.external_id}`);
  console.log(`[DB Check] Facts stored: ${factRows.length}`);
  for (const f of factRows)
    console.log(`  - [entity:${f.entity_id}] "${f.content}" (embedding: ${f.emb_bytes} bytes)`);
  if (factRows.length === 0) {
    console.warn(
      '[DB Check] No facts were written — the augmentation write did not persist anything.'
    );
  }

  console.log('\n🧠 5. Testing Recall...');
  const query = 'Do you remember what my name is and what food I like?';

  const recalled = await mem.recall(query);
  if (recalled.length === 0) {
    console.log('   [Recall] No memories found.');
  } else {
    console.log(`   [Recall] ${recalled.length} fact(s) retrieved:`);
    for (const fact of recalled) {
      const score = fact.score.toFixed(3);
      const date = fact.dateCreated ? ` (${fact.dateCreated})` : '';
      console.log(`     • [${score}] ${fact.content}${date}`);
    }
  }

  const response2 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: query }],
  });
  console.log(`🤖 AI: ${response2.choices[0].message.content}\n`);

  // Ensure the second message's augmentation also finishes before we close the DB
  await mem.engine.waitForAugmentation();

  console.log('扫 6. Cleaning up...');
  await mem.config.storage.close();
  console.log("✅ Test Complete! Check your folder for 'memori-test.db'.");
}

runLocalTest().catch((err: unknown) => {
  console.error('Test failed:', err);
});
