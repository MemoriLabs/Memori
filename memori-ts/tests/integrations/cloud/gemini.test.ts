import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { GoogleGenAI } from '@google/genai';
import { Memori } from '../../../src/memori.js';
import {
  clearMockState,
  setupMemoriMock,
  waitForPayload,
  assertPayloadIsValid,
  mockConfig,
} from './cloud-helpers.js';

// Automatically skip the test suite if API keys are missing from the .env
const hasKeys = !!(process.env.GEMINI_API_KEY && process.env.MEMORI_API_KEY);

describe.runIf(hasKeys)('Cloud Integration: Gemini', () => {
  const MODEL = 'gemini-2.5-flash';
  const TEST_PROMPT = "Say 'hello' in one word.";
  const PROVIDER = 'gemini';

  let memoriClient: Memori;
  let e_id: string;
  let p_id: string;

  beforeEach(() => {
    clearMockState();
    setupMemoriMock();

    // Dynamically grab the test name to isolate IDs
    const testName = expect.getState().currentTestName?.replace(/[^a-zA-Z0-9]/g, '-') || 'test';
    e_id = `user-${testName}`;
    p_id = `process-${testName}`;

    memoriClient = new Memori().attribution(e_id, p_id);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should handle standard generation', async () => {
    const geminiClient = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
    memoriClient.llm.register(geminiClient);

    const response = await geminiClient.models.generateContent({
      model: MODEL,
      contents: TEST_PROMPT,
    });

    await waitForPayload();
    assertPayloadIsValid(response.text!, e_id, p_id, PROVIDER);
  });

  it('should handle streaming generation', async () => {
    const geminiClient = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
    memoriClient.llm.register(geminiClient);

    const stream = await geminiClient.models.generateContentStream({
      model: MODEL,
      contents: TEST_PROMPT,
    });

    let fullContent = '';
    for await (const chunk of stream) {
      if (chunk.text) {
        fullContent += chunk.text;
      }
    }

    await waitForPayload();
    assertPayloadIsValid(fullContent, e_id, p_id, PROVIDER);
  });

  it('should handle multi-turn conversations', async () => {
    const geminiClient = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
    memoriClient.llm.register(geminiClient);

    // Turn 1
    await geminiClient.models.generateContent({
      model: MODEL,
      contents: 'My favorite color is blue.',
    });
    await waitForPayload(2);

    // Turn 2
    const response2 = await geminiClient.models.generateContent({
      model: MODEL,
      contents: 'What is my favorite color?',
    });

    // Validates that Memori stripped the injected history payload cleanly
    await waitForPayload(2);

    const content = response2.text!;
    expect(content.toLowerCase()).toContain('blue');

    assertPayloadIsValid(content, e_id, p_id, PROVIDER, 2);
  });

  it('should inject recall facts', async () => {
    const geminiClient = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY });
    memoriClient.llm.register(geminiClient);

    // Trip the manual mock injection state
    mockConfig.injectedFact = "The user's favorite word is 'MEMORI_42'.";

    const response = await geminiClient.models.generateContent({
      model: MODEL,
      contents: 'What is my favorite word?',
    });

    await waitForPayload();

    const content = response.text!;
    expect(content).toContain('MEMORI_42');

    assertPayloadIsValid('MEMORI_42', e_id, p_id, PROVIDER);
  });
});
