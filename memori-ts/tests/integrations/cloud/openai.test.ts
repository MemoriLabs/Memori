import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { OpenAI } from 'openai';
import { Memori } from '../../../src/memori.js';
import {
  clearMockState,
  setupMemoriMock,
  waitForPayload,
  assertPayloadIsValid,
  mockConfig,
} from './cloud-helpers.js';

const hasKeys = !!(process.env.OPENAI_API_KEY && process.env.MEMORI_API_KEY);

describe.runIf(hasKeys)('Cloud Integration: OpenAI', () => {
  const MODEL = 'gpt-4o-mini';
  const TEST_PROMPT = "Say 'hello' in one word.";
  const PROVIDER = 'openai';

  let memoriClient: Memori;
  let e_id: string;
  let p_id: string;

  beforeEach(() => {
    clearMockState();
    setupMemoriMock();

    // Generate a safe string from the current test name
    const testName = expect.getState().currentTestName?.replace(/[^a-zA-Z0-9]/g, '-') || 'test';
    e_id = `user-${testName}`;
    p_id = `process-${testName}`;

    memoriClient = new Memori().attribution(e_id, p_id);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should handle standard generation', async () => {
    const openaiClient = new OpenAI();
    memoriClient.llm.register(openaiClient);

    const response = await openaiClient.chat.completions.create({
      model: MODEL,
      messages: [{ role: 'user', content: TEST_PROMPT }],
    });

    await waitForPayload();
    assertPayloadIsValid(response.choices[0].message.content!, e_id, p_id, PROVIDER);
  });

  it('should handle streaming generation', async () => {
    const openaiClient = new OpenAI();
    memoriClient.llm.register(openaiClient);

    const stream = await openaiClient.chat.completions.create({
      model: MODEL,
      messages: [{ role: 'user', content: TEST_PROMPT }],
      stream: true,
    });

    let fullContent = '';
    for await (const chunk of stream) {
      fullContent += chunk.choices[0]?.delta?.content || '';
    }

    await waitForPayload();
    assertPayloadIsValid(fullContent, e_id, p_id, PROVIDER);
  });

  it('should handle multi-turn conversations', async () => {
    const openaiClient = new OpenAI();
    memoriClient.llm.register(openaiClient);

    await openaiClient.chat.completions.create({
      model: MODEL,
      messages: [{ role: 'user', content: 'My favorite color is blue.' }],
    });
    await waitForPayload(2);

    const response2 = await openaiClient.chat.completions.create({
      model: MODEL,
      messages: [{ role: 'user', content: 'What is my favorite color?' }],
    });
    // Waits for the payload that was appropriately stripped of injected history
    await waitForPayload(2);

    const content = response2.choices[0].message.content!;
    expect(content.toLowerCase()).toContain('blue');

    assertPayloadIsValid(content, e_id, p_id, PROVIDER, 2);
  });

  it('should inject recall facts', async () => {
    const openaiClient = new OpenAI();
    memoriClient.llm.register(openaiClient);

    // This dynamically trips the behavior inside our helper mock!
    mockConfig.injectedFact = "The user's favorite word is 'MEMORI_42'.";

    const response = await openaiClient.chat.completions.create({
      model: MODEL,
      messages: [{ role: 'user', content: 'What is my favorite word?' }],
    });

    await waitForPayload();

    const content = response.choices[0].message.content!;
    expect(content).toContain('MEMORI_42');

    assertPayloadIsValid('MEMORI_42', e_id, p_id, PROVIDER);
  });
});
