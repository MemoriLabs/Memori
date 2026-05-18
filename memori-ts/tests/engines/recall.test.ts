import { describe, it, expect, vi, beforeEach } from 'vitest';
import { RecallEngine } from '../../src/engines/recall.js';
import { Api } from '../../src/core/network.js';
import { Config } from '../../src/core/config.js';
import { SessionManager } from '../../src/core/session.js';
import { ProjectManager } from '../../src/core/project.js';
import { NativeEngine } from '../../src/core/engine.js';
import { LLMRequest } from '@memorilabs/axon';

describe('RecallEngine', () => {
  let recallEngine: RecallEngine;
  let mockApi: Api;
  let mockConfig: Config;
  let mockSession: SessionManager;
  let mockProject: ProjectManager;
  let mockNativeEngine: NativeEngine;

  beforeEach(() => {
    mockApi = { post: vi.fn(), get: vi.fn() } as unknown as Api;
    mockConfig = {
      entityId: 'test-entity',
      processId: 'test-process',
      recallRelevanceThreshold: 0.5,
    } as unknown as Config;
    mockSession = { id: 'test-session-id' } as unknown as SessionManager;
    mockProject = { id: 'test-project-id' } as unknown as ProjectManager;

    // 1. Create the mock native engine
    mockNativeEngine = {
      hasStorage: false,
      retrieve: vi.fn().mockReturnValue([]),
    } as unknown as NativeEngine;

    // 2. Pass it in as the SECOND argument!
    recallEngine = new RecallEngine(
      mockApi,
      mockNativeEngine,
      mockConfig,
      mockSession,
      mockProject
    );
  });

  describe('recall()', () => {
    it('should call API with correct payload when cloud is active', async () => {
      (mockApi.post as any).mockResolvedValue({ facts: ['fact1'] });

      const result = await recallEngine.recall('query');

      expect(result).toHaveLength(1);
      expect(mockApi.post).toHaveBeenCalled();
    });

    it('should call local Rust engine when storage is active', async () => {
      (mockNativeEngine as any).hasStorage = true;
      (mockNativeEngine.retrieve as any).mockReturnValue([
        { content: 'Local Fact', rank_score: 0.99, date_created: null },
      ]);

      const result = await recallEngine.recall('query');

      expect(result).toHaveLength(1);
      expect(result[0].content).toBe('Local Fact');
      expect(mockApi.post).not.toHaveBeenCalled();
    });
  });

  describe('handleRecall()', () => {
    it('should inject context into system prompt if facts are relevant', async () => {
      (mockApi.post as any).mockResolvedValue({
        facts: [{ id: 1, content: 'User likes apples', rank_score: 0.9 }],
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

    it('should sanitize malformed tool-call history returned by the API', async () => {
      (mockApi.post as any).mockResolvedValue({
        facts: [],
        messages: [
          { role: 'user', content: 'Weather in Tokyo?' },
          { role: 'assistant', content: '' },
          { role: 'tool', content: '{"temp": "21C"}' },
          { role: 'assistant', content: 'It is 21C.' },
          { role: 'model', content: 'Legacy model response.' },
        ],
      });

      const req = {
        messages: [{ role: 'user', content: 'What should I pack?' }],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);

      expect(newReq.messages).toEqual([
        { role: 'user', content: 'Weather in Tokyo?' },
        { role: 'assistant', content: 'It is 21C.' },
        { role: 'assistant', content: 'Legacy model response.' },
        { role: 'user', content: 'What should I pack?' },
      ]);
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
        facts: [{ id: 1, content: 'Fact', rank_score: 0.9 }],
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

    it('should include deduped summaries in the injected recall context', async () => {
      (mockApi.post as any).mockResolvedValue({
        facts: [
          {
            id: 1,
            content: 'User likes apples',
            rank_score: 0.9,
            summaries: [
              {
                content: 'User consistently mentions apples as a favorite.',
                date_created: '2023-01-01T12:00:00Z',
                entity_fact_id: 1,
                fact_id: 1,
              },
            ],
          },
          {
            id: 2,
            content: 'User buys apples weekly',
            rank_score: 0.8,
          },
        ],
        summaries: [
          {
            content: 'User consistently mentions apples as a favorite.',
            date_created: '2023-01-01T12:00:00Z',
            entity_fact_id: 2,
            fact_id: 2,
          },
          {
            content: 'User eats fruit regularly.',
            date_created: '2023-01-02T09:30:00Z',
            entity_fact_id: 2,
            fact_id: 2,
          },
        ],
      });

      const req = {
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'What fruit do I like?' },
        ],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);
      const systemMsg = newReq.messages.find((m) => m.role === 'system');

      expect(systemMsg?.content).toContain('## Summaries');
      expect(systemMsg?.content).toContain('[2023-01-01 12:00]');
      expect(systemMsg?.content).toContain('User eats fruit regularly.');
      expect(
        systemMsg?.content.match(/User consistently mentions apples as a favorite\./g)
      ).toHaveLength(1);
    });

    it('should return original request if no user message is found', async () => {
      // Empty messages array
      const req = { messages: [] } as unknown as LLMRequest;
      const newReq = await recallEngine.handleRecall(req, {} as any);
      expect(newReq).toBe(req);
    });

    it('should fetch from local Rust engine if storage is active', async () => {
      (mockNativeEngine as any).hasStorage = true;
      (mockNativeEngine as any).getConversationHistory = vi.fn().mockResolvedValue([]);
      (mockNativeEngine.retrieve as any).mockReturnValue([
        { content: 'Local storage memory', rank_score: 0.95, date_created: null },
      ]);

      const req = {
        messages: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'What do I like?' },
        ],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);

      const systemMsg = newReq.messages.find((m) => m.role === 'system');
      expect(systemMsg?.content).toContain('Local storage memory');
      expect(mockApi.post).not.toHaveBeenCalled();
    });

    it('should prepend conversation history from local storage when storage is active', async () => {
      const mockGetHistory = vi.fn().mockResolvedValue([
        { role: 'user', content: 'prior question' },
        { role: 'assistant', content: 'prior answer' },
      ]);
      (mockNativeEngine as any).hasStorage = true;
      (mockNativeEngine as any).getConversationHistory = mockGetHistory;
      (mockNativeEngine.retrieve as any).mockReturnValue([]);

      const req = {
        messages: [{ role: 'user', content: 'current question' }],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);

      expect(mockGetHistory).toHaveBeenCalledWith('test-session-id');
      expect(newReq.messages).toHaveLength(3);
      expect(newReq.messages[0]).toEqual({ role: 'user', content: 'prior question' });
      expect(newReq.messages[1]).toEqual({ role: 'assistant', content: 'prior answer' });
      expect(mockApi.post).not.toHaveBeenCalled();
    });

    it('should sanitize malformed tool-call history from local storage', async () => {
      const mockGetHistory = vi.fn().mockResolvedValue([
        { role: 'user', content: 'Weather in Tokyo?' },
        { role: 'assistant', content: '' },
        { role: 'tool', content: '{"temp": "21C"}' },
        { role: 'assistant', content: 'It is 21C.' },
      ]);
      (mockNativeEngine as any).hasStorage = true;
      (mockNativeEngine as any).getConversationHistory = mockGetHistory;
      (mockNativeEngine.retrieve as any).mockReturnValue([]);

      const req = {
        messages: [{ role: 'user', content: 'What should I pack?' }],
      } as unknown as LLMRequest;

      const newReq = await recallEngine.handleRecall(req, {} as any);

      expect(newReq.messages).toEqual([
        { role: 'user', content: 'Weather in Tokyo?' },
        { role: 'assistant', content: 'It is 21C.' },
        { role: 'user', content: 'What should I pack?' },
      ]);
    });
  });

  describe('agentRecall()', () => {
    it('calls GET agent/recall with entity and project params', async () => {
      (mockApi.get as any).mockResolvedValue({ facts: [] });
      await recallEngine.agentRecall();
      expect(mockApi.get).toHaveBeenCalledWith(expect.stringContaining('agent/recall'));
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('entity_id=test-entity');
      expect(url).toContain('project_id=test-project-id');
    });

    it('accepts explicit projectId and sessionId overrides', async () => {
      (mockApi.get as any).mockResolvedValue({ facts: [] });
      await recallEngine.agentRecall({ projectId: 'proj-override', sessionId: 'sess-override' });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('project_id=proj-override');
      expect(url).toContain('session_id=sess-override');
    });

    it('throws if sessionId provided without projectId', async () => {
      // Force project.id to be falsy
      (mockProject as any).id = undefined;
      await expect(recallEngine.agentRecall({ sessionId: 'some-session' })).rejects.toThrow(
        'sessionId cannot be provided without projectId'
      );
      (mockProject as any).id = 'test-project-id';
    });

    it('includes ISO string date params in the query string', async () => {
      (mockApi.get as any).mockResolvedValue({ facts: [] });
      const iso = '2024-06-15T00:00:00.000Z';
      await recallEngine.agentRecall({ dateStart: iso, dateEnd: iso });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('date_start=2024-06-15T00%3A00%3A00.000Z');
    });

    it('omits null and empty string params from query string', async () => {
      (mockApi.get as any).mockResolvedValue({ facts: [] });
      await recallEngine.agentRecall({ signal: undefined, source: '' });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).not.toContain('signal=');
      expect(url).not.toContain('source=');
    });
  });

  describe('agentRecallSummary()', () => {
    it('calls GET agent/recall/summary', async () => {
      (mockApi.get as any).mockResolvedValue({ summaries: [] });
      await recallEngine.agentRecallSummary();
      expect(mockApi.get).toHaveBeenCalledWith(expect.stringContaining('agent/recall/summary'));
    });

    it('throws if sessionId provided without projectId', async () => {
      (mockProject as any).id = undefined;
      await expect(recallEngine.agentRecallSummary({ sessionId: 'sess' })).rejects.toThrow(
        'sessionId cannot be provided without projectId'
      );
      (mockProject as any).id = 'test-project-id';
    });
  });

  describe('agentCompaction()', () => {
    const mockCompactionResponse = {
      continuation: { last_action: 'completed task A', next_expected_action: 'start task B' },
      environment: ['NODE_ENV=production'],
      messages: [{ content: 'hello', role: 'user', type: 'text' }],
      metadata: {
        date: { execution: '2024-01-01T00:00:00.000Z' },
        filter: { project: { id: 'test-project-id' } },
      },
      standing_orders: ['always format code before committing'],
      state: {
        active_tasks: ['refactor auth module'],
        open_loops: ['waiting for DB migration approval'],
        pending_results: [],
      },
      workspace_changes: ['src/auth/index.ts updated'],
    };

    it('calls GET agent/compaction with the correct route', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      await recallEngine.agentCompaction({ projectId: 'proj-1' });
      expect(mockApi.get).toHaveBeenCalledWith(expect.stringContaining('agent/compaction'));
    });

    it('includes project_id in the query string', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      await recallEngine.agentCompaction({ projectId: 'proj-1' });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('project_id=proj-1');
    });

    it('defaults projectId to the current project context when not supplied', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      await recallEngine.agentCompaction();
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('project_id=test-project-id');
    });

    it('includes session_id in the query string when provided', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      await recallEngine.agentCompaction({ projectId: 'proj-1', sessionId: 'sess-1' });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('session_id=sess-1');
    });

    it('includes num_messages in the query string when provided', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      await recallEngine.agentCompaction({ projectId: 'proj-1', numMessages: 10 });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('num_messages=10');
    });

    it('omits num_messages from the query string when not provided', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      await recallEngine.agentCompaction({ projectId: 'proj-1' });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).not.toContain('num_messages=');
    });

    it('returns the full compaction response', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      const result = await recallEngine.agentCompaction({ projectId: 'proj-1' });
      expect(result).toEqual(mockCompactionResponse);
    });

    it('throws when projectId is absent from both params and project context', async () => {
      (mockProject as any).id = undefined;
      await expect(recallEngine.agentCompaction()).rejects.toThrow(
        'projectId is required for agent compaction'
      );
      (mockProject as any).id = 'test-project-id';
    });

    it('accepts an explicit projectId override', async () => {
      (mockApi.get as any).mockResolvedValue(mockCompactionResponse);
      await recallEngine.agentCompaction({ projectId: 'override-proj' });
      const url: string = (mockApi.get as any).mock.calls[0][0];
      expect(url).toContain('project_id=override-proj');
    });
  });

  describe('recall() — local storage path error handling', () => {
    it('returns [] and warns when local retrieval throws', async () => {
      (mockNativeEngine as any).hasStorage = true;
      (mockNativeEngine.retrieve as any).mockRejectedValue(new Error('engine crash'));
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const result = await recallEngine.recall('query');
      expect(result).toEqual([]);
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });

    it('returns [] when storage is active but entityId is missing', async () => {
      (mockNativeEngine as any).hasStorage = true;
      (mockConfig as any).entityId = null;

      const result = await recallEngine.recall('query');
      expect(result).toEqual([]);
      (mockConfig as any).entityId = 'test-entity';
    });
  });

  describe('handleRecall() — local storage path error handling', () => {
    it('returns original request and warns when local retrieval throws', async () => {
      (mockNativeEngine as any).hasStorage = true;
      (mockNativeEngine.retrieve as any).mockRejectedValue(new Error('engine crash'));
      const consoleSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const req = { messages: [{ role: 'user', content: 'hi' }] } as any;
      const result = await recallEngine.handleRecall(req, {} as any);
      expect(result).toBe(req);
      consoleSpy.mockRestore();
    });

    it('returns original request when storage active but entityId missing', async () => {
      (mockNativeEngine as any).hasStorage = true;
      (mockConfig as any).entityId = null;

      const req = { messages: [{ role: 'user', content: 'hi' }] } as any;
      const result = await recallEngine.handleRecall(req, {} as any);
      expect(result).toBe(req);
      (mockConfig as any).entityId = 'test-entity';
    });
  });
});
