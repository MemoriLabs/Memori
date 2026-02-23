import 'dotenv/config';
import { OpenAI } from 'openai';
import { Memori } from '../../src/index.js';

// Environment check
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
  console.error('Error: OPENAI_API_KEY must be set in .env');
  process.exit(1);
}

// 1. Initialize the LLM Client
const client = new OpenAI({ apiKey: OPENAI_API_KEY });

// 2. Initialize Memori and Register the Client
const _memori = new Memori().llm.register(client).attribution('user-123', 'my-app');

async function main() {
  console.log('--- Step 1: Teaching the AI ---');
  const factPrompt = 'My favorite color is blue and I live in Paris.';
  console.log(`User: ${factPrompt}`);

  // This call automatically triggers Persistence and Augmentation in the background.
  const response1 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: factPrompt }],
  });

  console.log(`AI:   ${response1.choices[0].message.content}`);

  console.log('\n(Waiting 5 seconds for backend processing...)\n');
  await new Promise((resolve) => setTimeout(resolve, 5000));

  console.log('--- Step 2: Testing Recall ---');
  const questionPrompt = 'What is my favorite color?';
  console.log(`User: ${questionPrompt}`);

  // This call automatically triggers Recall, injecting the Paris/Blue facts into the prompt.
  const response2 = await client.chat.completions.create({
    model: 'gpt-4o-mini',
    messages: [{ role: 'user', content: questionPrompt }],
  });

  console.log(`AI:   ${response2.choices[0].message.content}`);
}

main().catch(console.error);
