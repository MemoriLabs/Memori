import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AugmentationEngine } from '../../src/engines/augmentation.js';
import { Api } from '../../src/core/network.js';
import { Config } from '../../src/core/config.js';
import { SessionManager } from '../../src/core/session.js';
import { LLMRequest, LLMResponse } from '@memorilabs/axon';

describe('AugmentationEngine', () => {
  let engine: AugmentationEngine;
  let mockApi: Api;
  let mockConfig: Config;
  let mockSession: SessionManager;

  beforeEach(() => {
    mockApi = { post: vi.fn().mockResolvedValue({}) } as unknown as Api;
    mockConfig = {
      entityId: 'u-1',
      processId: 'p-1',
      testMode: true,
    } as unknown as Config;
    mockSession = { id: 'sess-1' } as unknown as SessionManager;

    engine = new AugmentationEngine(mockApi, mockConfig, mockSession);
  });

  it('should trigger API post on handleAugmentation', async () => {
    const req = { messages: [{ role: 'user', content: 'learn this' }] } as unknown as LLMRequest;
    const res = { content: 'ok' } as LLMResponse;

    const mockCtx = {
      traceId: '123',
      startedAt: new Date(),
      metadata: {
        provider: 'openai',
        sdkVersion: '4.28.0',
        platform: null,
        framework: null,
      },
    } as any;

    await engine.handleAugmentation(req, res, mockCtx);

    expect(mockApi.post).toHaveBeenCalledWith(
      'cloud/augmentation',
      expect.objectContaining({
        conversation: expect.objectContaining({
          messages: [
            { role: 'user', content: 'learn this' },
            { role: 'assistant', content: 'ok' },
          ],
        }),
      })
    );
  });

  it('should log warning in testMode if API fails', async () => {
    (mockApi.post as any).mockRejectedValue(new Error('Augment fail'));
    const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

    const req = { messages: [{ role: 'user', content: 'hi' }] } as unknown as LLMRequest;
    const res = { content: 'ho' } as LLMResponse;

    const mockCtx = {
      traceId: '123',
      startedAt: new Date(),
      metadata: {
        provider: 'openai',
        sdkVersion: '4.28.0',
        platform: null,
        framework: null,
      },
    } as any;

    await engine.handleAugmentation(req, res, mockCtx);

    // Wait for the fire-and-forget promise
    await new Promise(process.nextTick);

    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });
});
