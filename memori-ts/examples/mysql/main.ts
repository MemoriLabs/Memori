/**
 * Quickstart: Memori + OpenAI + MySQL
 *
 * Demonstrates how Memori adds memory across conversations.
 */

import 'dotenv/config';
import mysql from 'mysql2/promise';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

const databaseConnectionString = process.env.DATABASE_CONNECTION_STRING;
if (!databaseConnectionString) {
  throw new Error('DATABASE_CONNECTION_STRING must be set in the environment');
}

const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const conn = await mysql.createConnection(databaseConnectionString);

const mem = new Memori({ conn }).llm.register(client);
mem.attribution('user-123', 'my-app');

if (!mem.config.storage) {
  throw new Error('Storage not initialized');
}

try {
  await mem.config.storage.build();

  console.log('You: My favorite food is pizza and I lived in new york');
  const response1 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'My favorite food is pizza and I lived in new york' }],
  });
  console.log(`AI: ${response1.choices[0]?.message?.content}\n`);

  await mem.engine.waitForAugmentation();

  console.log("You: What's my favorite food?");
  const response2 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: "What's my favorite food?" }],
  });
  console.log(`AI: ${response2.choices[0]?.message?.content}\n`);

  console.log('You: What city did I live in?');
  const response3 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'What city did I live in?' }],
  });
  console.log(`AI: ${response3.choices[0]?.message?.content}`);

  // Advanced Augmentation runs asynchronously to efficiently
  // create memories. For this example, a short lived command
  // line program, we need to wait for it to finish.
  await mem.engine.waitForAugmentation();
} finally {
  await mem.config.storage.close();
  await conn.end();
}
