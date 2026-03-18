import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import Anthropic from '@anthropic-ai/sdk';
import { Memori } from '../../../src/memori.js';
import {
  clearMockState,
  setupMemoriMock,
  waitForPayload,
  assertPayloadIsValid,
  mockConfig,
} from './cloud-helpers.js';

const hasKeys = !!(process.env.ANTHROPIC_API_KEY && process.env.MEMORI_API_KEY);

describe.runIf(hasKeys)('Cloud Integration: Anthropic', () => {
  const MODEL = 'claude-haiku-4-5';
  const TEST_PROMPT = "Say 'hello' in one word.";
  const PROVIDER = 'anthropic';

  let memoriClient: Memori;
  let e_id: string;
  let p_id: string;

  beforeEach(() => {
    clearMockState();
    setupMemoriMock();

    const testName = expect.getState().currentTestName?.replace(/[^a-zA-Z0-9]/g, '-') || 'test';
    e_id = `user-${testName}`;
    p_id = `process-${testName}`;

    memoriClient = new Memori().attribution(e_id, p_id);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should handle standard generation', async () => {
    const anthropicClient = new Anthropic();
    memoriClient.llm.register(anthropicClient);

    const response = await anthropicClient.messages.create({
      model: MODEL,
      max_tokens: 100,
      messages: [{ role: 'user', content: TEST_PROMPT }],
    });

    await waitForPayload();
    assertPayloadIsValid((response.content[0] as any).text, e_id, p_id, PROVIDER);
  });

  it('should handle streaming generation', async () => {
    const anthropicClient = new Anthropic();
    memoriClient.llm.register(anthropicClient);

    const stream = await anthropicClient.messages.create({
      model: MODEL,
      max_tokens: 100,
      messages: [{ role: 'user', content: TEST_PROMPT }],
      stream: true,
    });

    let fullContent = '';
    for await (const chunk of stream) {
      if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
        fullContent += chunk.delta.text;
      }
    }

    await waitForPayload();
    assertPayloadIsValid(fullContent, e_id, p_id, PROVIDER);
  });

  it('should handle multi-turn conversations', async () => {
    const anthropicClient = new Anthropic();
    memoriClient.llm.register(anthropicClient);

    await anthropicClient.messages.create({
      model: MODEL,
      max_tokens: 100,
      messages: [{ role: 'user', content: 'My favorite color is blue.' }],
    });
    await waitForPayload(2);

    const response2 = await anthropicClient.messages.create({
      model: MODEL,
      max_tokens: 100,
      messages: [{ role: 'user', content: 'What is my favorite color?' }],
    });
    await waitForPayload(2);

    const content = (response2.content[0] as any).text;
    expect(content.toLowerCase()).toContain('blue');

    assertPayloadIsValid(content, e_id, p_id, PROVIDER, 2);
  });

  it('should inject recall facts', async () => {
    const anthropicClient = new Anthropic();
    memoriClient.llm.register(anthropicClient);

    mockConfig.injectedFact = "The user's favorite word is 'MEMORI_42'.";

    const response = await anthropicClient.messages.create({
      model: MODEL,
      max_tokens: 100,
      messages: [{ role: 'user', content: 'What is my favorite word?' }],
    });

    await waitForPayload();

    const content = (response.content[0] as any).text;
    expect(content).toContain('MEMORI_42');

    assertPayloadIsValid('MEMORI_42', e_id, p_id, PROVIDER);
  });
});
