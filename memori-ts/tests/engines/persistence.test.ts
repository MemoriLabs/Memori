import { describe, it, expect, vi, beforeEach } from 'vitest';
import { PersistenceEngine } from '../../src/engines/persistence.js';
import { Api } from '../../src/core/network.js';
import { Config } from '../../src/core/config.js';
import { SessionManager } from '../../src/core/session.js';
import { LLMRequest, LLMResponse } from '@memorilabs/axon';

describe('PersistenceEngine', () => {
  let engine: PersistenceEngine;
  let mockApi: Api;
  let mockConfig: Config;
  let mockSession: SessionManager;

  beforeEach(() => {
    mockApi = { post: vi.fn().mockResolvedValue({}) } as unknown as Api;
    mockConfig = { entityId: 'u-1', processId: 'p-1' } as unknown as Config;
    mockSession = { id: 'sess-1' } as unknown as SessionManager;

    engine = new PersistenceEngine(mockApi, mockConfig, mockSession);
  });

  it('should return response immediately if no session ID', async () => {
    (mockSession as any).id = undefined;
    const req = { messages: [] } as unknown as LLMRequest;
    const res = { content: 'response' } as LLMResponse;

    await engine.handlePersistence(req, res, {} as any);
    expect(mockApi.post).not.toHaveBeenCalled();
  });

  it('should post conversation to API if valid user message exists', async () => {
    const req = {
      messages: [
        { role: 'system', content: 'sys' },
        { role: 'user', content: 'hello' },
      ],
    } as unknown as LLMRequest;
    const res = { content: 'world' } as LLMResponse;

    await engine.handlePersistence(req, res, {} as any);

    expect(mockApi.post).toHaveBeenCalledWith(
      'cloud/conversation/messages',
      expect.objectContaining({
        session: { id: 'sess-1' },
        messages: [
          { role: 'user', type: 'text', text: 'hello' },
          { role: 'assistant', type: 'text', text: 'world' },
        ],
      })
    );
  });

  it('should handle API errors gracefully (no throw)', async () => {
    (mockApi.post as any).mockRejectedValue(new Error('fail'));
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const req = { messages: [{ role: 'user', content: 'hi' }] } as unknown as LLMRequest;
    const res = { content: 'ho' } as LLMResponse;

    await expect(engine.handlePersistence(req, res, {} as any)).resolves.toEqual(res);
    expect(consoleSpy).toHaveBeenCalled();
  });

  it('should not post if no user message is found in history', async () => {
    const req = {
      messages: [{ role: 'system', content: 'sys' }],
    } as unknown as LLMRequest;
    const res = { content: 'resp' } as LLMResponse;

    await engine.handlePersistence(req, res, {} as any);
    expect(mockApi.post).not.toHaveBeenCalled();
  });
});
