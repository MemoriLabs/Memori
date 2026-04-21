import 'dotenv/config';
import mysql from 'mysql2/promise';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

async function runMysqlTest() {
  console.log('🚀 1. Initializing Memori with local MySQL...');
  const conn = await mysql.createConnection({
    host: 'localhost',
    user: 'memori',
    database: 'memori_test',
    password: 'memori',
    port: 3307,
  });

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const mem = new Memori({ conn }).llm.register(client);
  mem.attribution('mysql-user', 'mysql-test');

  console.log('🧱 2. Building Database Schema...');
  if (!mem.config.storage) throw new Error('Storage not initialized');
  await mem.config.storage.build();

  console.log('\n💬 3. Sending teaching message...');
  await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'I am allergic to peanuts.' }],
  });

  console.log('\n⏳ 4. Waiting for Rust engine...');
  await mem.engine.waitForAugmentation();

  // Verification
  const [eRows] = await conn.execute('SELECT id, external_id FROM memori_entity');
  const [fRows] = await conn.execute(
    'SELECT id, entity_id, content, LENGTH(content_embedding) as emb FROM memori_entity_fact'
  );
  console.log(
    `\n[DB Check] Entities: ${(eRows as unknown[]).length}, Facts: ${(fRows as unknown[]).length}`
  );

  console.log('\n🧠 5. Testing Recall...');
  const query = "Is there anything I shouldn't eat?";
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

  const response = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: query }],
  });
  console.log(`\n🤖 AI: ${response.choices[0].message.content}`);

  await mem.engine.waitForAugmentation();

  console.log('6. Cleaning up...');
  await mem.config.storage.close();

  console.log("✅ Test Complete! Check your folder for 'memori-test.db'.");
}

runMysqlTest().catch(console.error);
