import 'dotenv/config';
import { DataSource } from 'typeorm';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

async function runTypeOrmTest() {
  console.log('🚀 1. Initializing Memori with TypeORM (SQLite)...');

  const dataSource = new DataSource({
    type: 'better-sqlite3',
    database: 'memori-typeorm.db',
  });
  await dataSource.initialize();

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  // Pass the TypeORM DataSource to Memori
  const mem = new Memori({ conn: dataSource }).llm.register(client);
  mem.attribution('typeorm-user', 'typeorm-test');

  console.log('🧱 2. Building Database Schema via TypeORM Adapter...');
  if (!mem.config.storage) throw new Error('Storage not initialized');
  await mem.config.storage.build();

  console.log('\n💬 3. Sending teaching message...');
  await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'I have a pet dog named Barnaby.' }],
  });

  console.log('\n⏳ 4. Waiting for Rust engine...');
  await mem.engine.waitForAugmentation();

  // Verification using TypeORM's query runner
  const entities = await dataSource.query<unknown[]>('SELECT id, external_id FROM memori_entity');
  const facts = await dataSource.query<unknown[]>(
    'SELECT id, entity_id, content FROM memori_entity_fact'
  );

  console.log(`\n[DB Check] Entities: ${entities.length}, Facts: ${facts.length}`);

  console.log('\n🧠 5. Testing Recall...');
  const query = 'What is the name of my pet?';
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
  await mem.config.storage.close(); // Drops the QueryRunner
  await dataSource.destroy();
  console.log('✅ TypeORM Test Complete!');
}

runTypeOrmTest().catch(console.error);
