import 'dotenv/config';
import pg from 'pg';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

async function runPostgresTest() {
  console.log('🚀 1. Initializing Memori with local PostgreSQL...');

  // =========================================================================
  // FIX: Force-create the database so we don't rely on Docker's boot scripts
  // =========================================================================
  const setupPool = new pg.Pool({
    host: 'localhost',
    user: 'memori',
    password: 'memori',
    database: 'memori_test',
    port: 5432,
  });

  try {
    await setupPool.query('CREATE DATABASE memori_test;');
    console.log('   [Setup] Successfully auto-created "memori_test" database.');
  } catch (err: unknown) {
    const pgErr = err as { code?: string; message?: string };
    if (pgErr.code === '42P04') {
      // 42P04 is the Postgres code for "database already exists" - this is fine!
    } else {
      console.error('\n🚨 CRITICAL DB ERROR 🚨');
      console.error('If you see "role does not exist" or "password authentication failed",');
      console.error('your Mac has a background Postgres app hijacking port 5432!');
      console.error('Error Details:', pgErr.message, '\n');
    }
  } finally {
    await setupPool.end();
  }
  // =========================================================================

  // Now connect to the actual database
  const pool = new pg.Pool({
    host: 'localhost',
    user: 'memori',
    database: 'memori_test',
    password: 'memori',
    port: 5432,
  });

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
  const mem = new Memori({ conn: pool }).llm.register(client);
  mem.attribution('pg-user', 'pg-test');

  console.log('🧱 2. Building Database Schema...');
  if (!mem.config.storage) throw new Error('Storage not initialized');
  await mem.config.storage.build();

  console.log('\n💬 3. Sending teaching message...');
  await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'My favorite color is blue.' }],
  });

  console.log('\n⏳ 4. Waiting for Rust engine...');
  await mem.engine.waitForAugmentation();

  // Verification
  const eRes = await pool.query('SELECT id, external_id FROM memori_entity');
  const fRes = await pool.query(
    'SELECT id, entity_id, content, OCTET_LENGTH(content_embedding) as emb FROM memori_entity_fact'
  );
  console.log(`\n[DB Check] Entities: ${eRes.rows.length}, Facts: ${fRes.rows.length}`);

  console.log('\n🧠 5. Testing Recall...');
  const query = 'What is my favorite color?';
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

  console.log('\n🧹 6. Cleaning up...');
  await mem.config.storage.close();
  console.log('✅ Test Complete!');
}

runPostgresTest().catch(console.error);
