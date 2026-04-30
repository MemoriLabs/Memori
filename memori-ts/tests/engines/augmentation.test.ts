import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AugmentationEngine } from '../../src/engines/augmentation.js';
import { Api } from '../../src/core/network.js';
import { Config } from '../../src/core/config.js';
import { SessionManager } from '../../src/core/session.js';
import { ProjectManager } from '../../src/core/project.js';
import { NativeEngine } from '../../src/core/engine.js';
import { LLMRequest, LLMResponse } from '@memorilabs/axon';

describe('AugmentationEngine', () => {
  let engine: AugmentationEngine;
  let mockApi: Api;
  let mockCollectorApi: Api;
  let mockConfig: Config;
  let mockSession: SessionManager;
  let mockProject: ProjectManager;
  let mockNativeEngine: NativeEngine;

  beforeEach(() => {
    mockApi = { post: vi.fn().mockResolvedValue({}) } as unknown as Api;
    mockCollectorApi = { post: vi.fn().mockResolvedValue({}) } as unknown as Api;
    mockConfig = {
      entityId: 'u-1',
      processId: 'p-1',
      testMode: true,
    } as unknown as Config;
    mockSession = { id: 'sess-1' } as unknown as SessionManager;
    mockProject = { id: 'proj-1' } as unknown as ProjectManager;
    mockNativeEngine = {
      hasStorage: false,
      submitAugmentation: vi.fn(),
    } as unknown as NativeEngine;

    engine = new AugmentationEngine(
      mockApi,
      mockCollectorApi,
      mockNativeEngine,
      mockConfig,
      mockSession,
      mockProject
    );
  });

  describe('handleAugmentation()', () => {
    it('should trigger API post on handleAugmentation', async () => {
      const req = {
        messages: [{ role: 'user', content: 'learn this', type: 'text' }],
      } as unknown as LLMRequest;
      const res = { content: 'ok', type: 'text' } as LLMResponse;

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

      expect(mockCollectorApi.post).toHaveBeenCalledWith(
        'cloud/augmentation',
        expect.objectContaining({
          conversation: expect.objectContaining({
            messages: [
              { role: 'user', content: 'learn this', type: 'text' },
              { role: 'assistant', content: 'ok', type: 'text' },
            ],
          }),
        })
      );
    });

    it('should trigger Rust local engine if storage is active', async () => {
      (mockNativeEngine as any).hasStorage = true;

      const req = {
        messages: [{ role: 'user', content: 'learn this', type: 'text' }],
      } as unknown as LLMRequest;
      const res = { content: 'ok', type: 'text' } as LLMResponse;
      const mockCtx = { metadata: {} } as any;

      await engine.handleAugmentation(req, res, mockCtx);

      expect(mockApi.post).not.toHaveBeenCalled();
      expect(mockNativeEngine.submitAugmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          conversation_messages: [
            { role: 'user', content: 'learn this', type: 'text' },
            { role: 'assistant', content: 'ok', type: 'text' },
          ],
        })
      );
    });

    it('should log warning in testMode if API fails', async () => {
      (mockCollectorApi.post as any).mockRejectedValue(new Error('Augment fail'));
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const req = {
        messages: [{ role: 'user', content: 'hi', type: 'text' }],
      } as unknown as LLMRequest;
      const res = { content: 'ho', type: 'text' } as LLMResponse;

      const mockCtx = {
        traceId: '123',
        startedAt: new Date(),
        metadata: {},
      } as any;

      await engine.handleAugmentation(req, res, mockCtx);
      await new Promise(process.nextTick);

      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });
  });

  describe('handleAgentAugmentation()', () => {
    const req = {
      messages: [{ role: 'user', content: 'hello', type: 'text' }],
    } as unknown as LLMRequest;
    const res = { content: 'world', type: 'text' } as LLMResponse;
    const mockCtx = {
      traceId: '123',
      startedAt: new Date(),
      metadata: { provider: 'openai', sdkVersion: null, platform: 'openclaw' },
    } as any;

    it('should place attribution at the top level of the augmentation payload', async () => {
      await engine.handleAgentAugmentation(req, res, mockCtx);
      await new Promise(process.nextTick);

      const augCall = (mockCollectorApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/augmentation'
      );
      expect(augCall[1].attribution).toEqual({
        entity: { id: 'u-1' },
        process: { id: 'p-1' },
      });
    });

    it('should not include attribution inside meta', async () => {
      await engine.handleAgentAugmentation(req, res, mockCtx);
      await new Promise(process.nextTick);

      const augCall = (mockCollectorApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/augmentation'
      );
      expect(augCall[1].meta).not.toHaveProperty('attribution');
    });

    it('should include project at the top level of the augmentation payload', async () => {
      await engine.handleAgentAugmentation(req, res, mockCtx);
      await new Promise(process.nextTick);

      const augCall = (mockCollectorApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/augmentation'
      );
      expect(augCall[1].project).toEqual({ id: 'proj-1' });
    });

    it('should nest summary inside session', async () => {
      await engine.handleAgentAugmentation(req, res, mockCtx, null, 'turn summary');
      await new Promise(process.nextTick);

      const augCall = (mockCollectorApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/augmentation'
      );
      expect(augCall[1].session).toEqual({ id: 'sess-1', summary: 'turn summary' });
    });

    it('should default summary to null when not provided', async () => {
      await engine.handleAgentAugmentation(req, res, mockCtx);
      await new Promise(process.nextTick);

      const augCall = (mockCollectorApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/augmentation'
      );
      expect(augCall[1].session.summary).toBeNull();
    });

    it('should also post to agent/conversation/turn', async () => {
      await engine.handleAgentAugmentation(req, res, mockCtx);
      await new Promise(process.nextTick);

      const turnCall = (mockApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/conversation/turn'
      );
      expect(turnCall).toBeDefined();
      expect(turnCall[1]).toMatchObject({
        attribution: { entity: { id: 'u-1' }, process: { id: 'p-1' } },
        project: { id: 'proj-1' },
        session: { id: 'sess-1' },
      });
    });

    it('should attach trace to the assistant message in the turn payload', async () => {
      const trace = { tools: [{ name: 'search', args: { q: 'test' }, result: 'found' }] };

      await engine.handleAgentAugmentation(req, res, mockCtx, trace);
      await new Promise(process.nextTick);

      const turnCall = (mockApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/conversation/turn'
      );
      expect(turnCall[1].messages).toEqual([
        { role: 'user', content: 'hello', type: 'text', trace: null },
        { role: 'assistant', content: 'world', type: 'text', trace },
      ]);
    });

    it('should set trace to null on both messages when no trace is provided', async () => {
      await engine.handleAgentAugmentation(req, res, mockCtx);
      await new Promise(process.nextTick);

      const turnCall = (mockApi.post as any).mock.calls.find(
        (c: any) => c[0] === 'agent/conversation/turn'
      );
      expect(turnCall[1].messages[0].trace).toBeNull();
      expect(turnCall[1].messages[1].trace).toBeNull();
    });

    it('should not call cloud endpoints when BYODB storage is active', async () => {
      (mockNativeEngine as any).hasStorage = true;

      await engine.handleAgentAugmentation(req, res, mockCtx);

      expect(mockApi.post).not.toHaveBeenCalled();
      expect(mockNativeEngine.submitAugmentation).toHaveBeenCalled();
    });
  });
});
