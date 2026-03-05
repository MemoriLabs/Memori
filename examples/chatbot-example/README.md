# Memori Next.js Chatbot Example

Next.js chatbot using Vercel AI SDK and Memori for memory. The assistant recalls and stores context across turns using the new `recordTurn()` method.

## Prerequisites

- Node 18+

## Setup

```bash
cp .env.example .env.local
```

Set `OPENAI_API_KEY` (required). Optionally set `MEMORI_API_KEY` and `MEMORI_TEST_MODE=1` for Memori Cloud staging.

```bash
npm install
npm run dev
```

Open http://localhost:3000. Use "New conversation" to start a fresh session; the current session is used for Memori recall and persistence.

## How it works

This example demonstrates the new `recordTurn()` method for non-Axon flows:

1. **Recall**: Uses `mem.recall(userMessage)` to fetch relevant memories before streaming
2. **Stream**: Uses Vercel AI SDK `streamText()` for the LLM response  
3. **Record**: Uses `mem.recordTurn(userMessage, assistantResponse, { model })` in the `onFinish` callback to persist the turn

This enables full Memori functionality (recall + persistence + augmentation) with modern streaming frameworks.