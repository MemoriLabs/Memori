import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RecallEngine } from '../../src/engines/recall.js';
import { Api } from '../../src/core/network.js';
import { Config } from '../../src/core/config.js';
import { SessionManager } from '../../src/core/session.js';
import { LLMRequest } from '@memorilabs/axon';

describe('RecallEngine', () => {
  let recallEngine: RecallEngine;
  let mockApi: Api;
  let mockConfig: Config;
  let mockSession: SessionManager;

  beforeEach(() => {
    mockApi = { post: vi.fn() } as unknown as Api;
    mockConfig = {
      entityId: 'test-entity',
      processId: 'test-process',
      recallRelevanceThreshold: 0.5,
    } as unknown as Config;
    mockSession = { id: 'test-session-id' } as unknown as SessionManager;

    recallEngine = new RecallEngine(mockApi, mockConfig, mockSession);
  });

  describe('recall()', () => {
    it('should call API with correct payload', async () => {
      (mockApi.post as any).mockResolvedValue({ facts: ['fact1'] });
      const result = await recallEngine.recall('query');
      expect(result).toHaveLength(1);
      expect(mockApi.post).toHaveBeenCalled();
    });
  });

  describe('handleRecall()', () => {
    it('should inject context into system prompt if facts are relevant', async () => {
      (mockApi.post as any).mockResolvedValue({
        facts: [{ content: 'User likes apples', rank_score: 0.9 }],
      });

      const req = {
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'What do I like?' },
        ],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);

      const systemMsg = newReq.messages.find((m) => m.role === 'system');
      expect(systemMsg?.content).toContain('User likes apples');
      expect(systemMsg?.content).toContain('<memori_context>');
    });

    it('should prepend history if API returns conversation history', async () => {
      (mockApi.post as any).mockResolvedValue({
        facts: [],
        messages: [
          { role: 'user', content: 'past msg' },
          { role: 'assistant', content: 'past answer' },
        ],
      });

      const req = {
        messages: [{ role: 'user', content: 'current msg' }],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);

      expect(newReq.messages).toHaveLength(3);
      expect(newReq.messages[0].content).toBe('past msg');
    });

    it('should fail silently and return original request on API error', async () => {
      (mockApi.post as any).mockRejectedValue(new Error('Network fail'));
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const req = { messages: [{ role: 'user', content: 'hi' }] } as unknown as LLMRequest;
      const newReq = await recallEngine.handleRecall(req, {} as any);

      expect(newReq).toBe(req);
      expect(consoleSpy).toHaveBeenCalled();

      consoleSpy.mockRestore();
    });

    it('should create a new system message if one does not exist', async () => {
      (mockApi.post as any).mockResolvedValue({
        facts: [{ content: 'Fact', rank_score: 0.9 }],
      });

      // Request WITHOUT a system message
      const req = {
        messages: [{ role: 'user', content: 'Query' }],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);

      // Verify a system message was added to the front
      expect(newReq.messages[0].role).toBe('system');
      expect(newReq.messages[0].content).toContain('Fact');
    });

    it('should return original request if no user message is found', async () => {
      // Empty messages array
      const req = { messages: [] } as unknown as LLMRequest;
      const newReq = await recallEngine.handleRecall(req, {} as any);
      expect(newReq).toBe(req);
    });
  });
});
