/**
 * Memori + Sequelize + MySQL
 *
 * Same narrative as examples/mysql/main.ts; shows passing a Sequelize instance.
 */

import 'dotenv/config';
import { Sequelize } from 'sequelize';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

const databaseConnectionString = process.env.DATABASE_CONNECTION_STRING;
if (!databaseConnectionString) {
  throw new Error('DATABASE_CONNECTION_STRING must be set in the environment');
}

const sequelize = new Sequelize(databaseConnectionString, {
  dialect: 'mysql',
  logging: false,
});

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const mem = new Memori({ conn: sequelize }).llm.register(client);
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
  await sequelize.close();
}
