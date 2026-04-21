import 'dotenv/config';
import { Sequelize } from 'sequelize';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

async function runSequelizeTest() {
  console.log('🚀 1. Initializing Memori with Sequelize (MySQL)...');

  const sequelize = new Sequelize('memori_test', 'memori', 'memori', {
    host: 'localhost',
    port: 3307,
    dialect: 'mysql',
    logging: false, // Turn off Sequelize's noisy console logs
  });

  const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

  // Pass the Sequelize instance to Memori
  const mem = new Memori({ conn: sequelize }).llm.register(client);
  mem.attribution('sequelize-user', 'sequelize-test');

  console.log('🧱 2. Building Database Schema via Sequelize Adapter...');
  if (!mem.config.storage) throw new Error('Storage not initialized');
  await mem.config.storage.build();

  console.log('\n💬 3. Sending teaching message...');
  await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'I drink 3 cups of coffee every morning.' }],
  });

  console.log('\n⏳ 4. Waiting for Rust engine...');
  await mem.engine.waitForAugmentation();

  // Verification using Sequelize's raw query method
  const [entities] = await sequelize.query('SELECT id, external_id FROM memori_entity');
  const [facts] = await sequelize.query('SELECT id, entity_id, content FROM memori_entity_fact');

  console.log(`\n[DB Check] Entities: ${entities.length}, Facts: ${facts.length}`);

  console.log('\n🧠 5. Testing Recall...');
  const query = 'How much coffee do I drink?';
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
  await sequelize.close();
  console.log('✅ Sequelize Test Complete!');
}

runSequelizeTest().catch(console.error);
