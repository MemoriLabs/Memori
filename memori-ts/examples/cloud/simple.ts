/**
 * Quickstart: Memori + OpenAI + Cloud
 *
 * Demonstrates how Memori adds memory across conversations.
 */

import 'dotenv/config';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

// Setup OpenAI
const client = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '<your_api_key_here>',
});

// Setup Memori - that's it!
const mem = new Memori().llm.register(client);
mem.attribution('user-123', 'my-app');

async function main() {
  // First conversation - establish facts
  console.log('You: My favorite color is blue and I live in Paris');
  const response1 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'My favorite color is blue and I live in Paris' }],
  });
  console.log(`AI: ${response1.choices[0]?.message?.content}\n`);

  // Give the cloud API a brief moment to index the new memory
  await new Promise((resolve) => setTimeout(resolve, 2000));

  // Second conversation - Memori recalls context automatically
  console.log("You: What's my favorite color?");
  const response2 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: "What's my favorite color?" }],
  });
  console.log(`AI: ${response2.choices[0]?.message?.content}\n`);

  // Third conversation - context is maintained
  console.log('You: What city do I live in?');
  const response3 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: 'What city do I live in?' }],
  });
  console.log(`AI: ${response3.choices[0]?.message?.content}\n`);

  // Advanced Augmentation runs asynchronously to efficiently
  // create memories. For this example, a short lived command
  // line program, we need to wait for it to finish.
  await new Promise((resolve) => setTimeout(resolve, 1000));
}

main().catch(console.error);
