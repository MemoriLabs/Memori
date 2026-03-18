import { vi, expect } from 'vitest';
import { Api } from '../../../src/core/network.js';

export interface MockMessage {
  role: string;
  text?: string;
  content?: string;
}

export interface MockPayload {
  messages?: MockMessage[];
  attribution?: { entity?: { id?: string }; process?: { id?: string } };
  meta?: { llm?: { model?: { provider?: string } } };
}

export const state = {
  captured_message_payloads: [] as MockPayload[],
  captured_augmentation_payloads: [] as MockPayload[],
  captured_recall_payloads: [] as MockPayload[],
  simulated_cloud_history: [] as MockMessage[],
};

export const mockConfig = {
  injectedFact: null as string | null,
};

export function clearMockState() {
  state.captured_message_payloads = [];
  state.captured_augmentation_payloads = [];
  state.captured_recall_payloads = [];
  state.simulated_cloud_history = [];
  mockConfig.injectedFact = null;
}

export function setupMemoriMock() {
  // Gracefully intercept all POST requests made by the Memori Api class
  vi.spyOn(Api.prototype, 'post').mockImplementation(async (route: string, body?: any) => {
    if (route.includes('cloud/conversation/messages')) {
      state.captured_message_payloads.push(body as MockPayload);
      const newMessages = body?.messages || [];
      for (const msg of newMessages) {
        // Prevent dupes in our mock history database
        if (!state.simulated_cloud_history.find((m) => JSON.stringify(m) === JSON.stringify(msg))) {
          state.simulated_cloud_history.push(msg);
        }
      }
      return {};
    } else if (route.includes('cloud/augmentation')) {
      state.captured_augmentation_payloads.push(body as MockPayload);
      return {};
    } else if (route.includes('cloud/recall')) {
      state.captured_recall_payloads.push(body as MockPayload);

      if (mockConfig.injectedFact) {
        return {
          messages: [],
          facts: [{ content: mockConfig.injectedFact, rank_score: 0.99 }],
        };
      }
      return { messages: [...state.simulated_cloud_history] };
    }
    return {};
  });
}

export async function waitForPayload(expectedLength = 2, timeoutMs = 3000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    if (state.captured_message_payloads.length > 0) {
      const lastPayload = state.captured_message_payloads.at(-1);
      const messages = lastPayload?.messages || [];
      if (messages.length >= expectedLength) {
        return;
      }
    }
    await new Promise((r) => setTimeout(r, 50));
  }

  const lastPayload = state.captured_message_payloads.at(-1);
  const len = lastPayload?.messages?.length || 0;
  throw new Error(
    `Timed out waiting for payload after ${timeoutMs}ms. Expected ${expectedLength} messages, got ${len}`
  );
}

export function assertPayloadIsValid(
  expectedContent: string,
  entityId: string,
  processId: string,
  expectedProvider: string,
  expectedHistoryLength = 2
) {
  const msgPayload = state.captured_message_payloads.at(-1);
  const augPayload = state.captured_augmentation_payloads.at(-1);
  const recallPayload = state.captured_recall_payloads.at(-1);

  expect(msgPayload).toBeDefined();

  const messages = msgPayload?.messages || [];
  expect(messages.length).toBeGreaterThanOrEqual(expectedHistoryLength);

  const userMessage = messages.findLast((m) => m.role === 'user');
  expect(userMessage).toBeDefined();

  const assistantMessage = messages.findLast((m) => ['assistant', 'model'].includes(m.role));
  expect(assistantMessage).toBeDefined();

  const actualText = (assistantMessage?.text || assistantMessage?.content || '').toLowerCase();
  expect(actualText).toContain(expectedContent.toLowerCase());

  expect(msgPayload?.attribution?.entity?.id).toBe(entityId);
  expect(msgPayload?.attribution?.process?.id).toBe(processId);

  if (augPayload) {
    const provider = augPayload.meta?.llm?.model?.provider;
    expect(provider).toBe(expectedProvider);
  }

  if (recallPayload) {
    expect(recallPayload.attribution?.entity?.id).toBe(entityId);
  }
}
