import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createMemoriCompactionTool } from '../../src/tools/memori-compaction.js';
import { QuotaExceededError } from '@memorilabs/memori';
import type { ToolDeps } from '../../src/tools/types.js';

vi.mock('../../src/utils/memori-client.js', () => ({
  createRecallClient: vi.fn(() => ({
    agentCompaction: vi.fn(async () => ({
      continuation: { last_action: 'ran tests', next_expected_action: 'review results' },
      environment: ['NODE_ENV=test'],
      messages: [],
      metadata: {
        date: { execution: '2024-01-01T00:00:00Z' },
        filter: { project: { id: 'proj-1' } },
      },
      standing_orders: [],
      state: { active_tasks: [], open_loops: [], pending_results: [] },
      workspace_changes: [],
    })),
  })),
}));

describe('tools/memori-compaction', () => {
  let deps: ToolDeps;

  beforeEach(() => {
    vi.clearAllMocks();

    deps = {
      api: {} as any,
      config: { apiKey: 'test-key', entityId: 'test-entity', projectId: 'default-project' },
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        section: vi.fn(),
        endSection: vi.fn(),
      } as any,
    };
  });

  describe('tool definition', () => {
    it('should have correct name and label', () => {
      const tool = createMemoriCompactionTool(deps);
      expect(tool.name).toBe('memori_compaction');
      expect(tool.label).toBe('Compact Agent Memory');
    });

    it('should require projectId', () => {
      const tool = createMemoriCompactionTool(deps);
      expect(tool.parameters.required).toContain('projectId');
    });

    it('should define expected parameter properties', () => {
      const tool = createMemoriCompactionTool(deps);
      const props = tool.parameters.properties;
      expect(props).toHaveProperty('projectId');
      expect(props).toHaveProperty('sessionId');
      expect(props).toHaveProperty('numMessages');
    });
  });

  describe('execute', () => {
    it('should call createRecallClient with apiKey and entityId', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriCompactionTool(deps);

      await tool.execute('call-1', { projectId: 'proj-1' });

      expect(createRecallClient).toHaveBeenCalledWith('test-key', 'test-entity');
    });

    it('should use config.projectId as fallback when not supplied', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriCompactionTool(deps);

      await tool.execute('call-1', {});

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentCompaction).toHaveBeenCalledWith(
        expect.objectContaining({ projectId: 'default-project' })
      );
    });

    it('should allow LLM-provided projectId to override config default', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriCompactionTool(deps);

      await tool.execute('call-1', { projectId: 'override-project' });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentCompaction).toHaveBeenCalledWith(
        expect.objectContaining({ projectId: 'override-project' })
      );
    });

    it('should pass sessionId through to agentCompaction', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriCompactionTool(deps);

      await tool.execute('call-1', { projectId: 'proj-1', sessionId: 'sess-1' });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentCompaction).toHaveBeenCalledWith(
        expect.objectContaining({ sessionId: 'sess-1' })
      );
    });

    it('should pass numMessages through when provided', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriCompactionTool(deps);

      await tool.execute('call-1', { projectId: 'proj-1', numMessages: 10 });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentCompaction).toHaveBeenCalledWith(
        expect.objectContaining({ numMessages: 10 })
      );
    });

    it('should return JSON-stringified result on success', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const mockResult = {
        continuation: { last_action: 'deployed', next_expected_action: 'monitor' },
        environment: ['ENV=prod'],
        messages: [{ content: 'hi', role: 'user', type: 'text' }],
        metadata: {
          date: { execution: '2024-06-01T00:00:00Z' },
          filter: { project: { id: 'p1' } },
        },
        standing_orders: ['keep it stable'],
        state: { active_tasks: ['deploy'], open_loops: ['verify health'], pending_results: [] },
        timeline: 'deployed to prod',
        workspace_changes: ['updated config.yaml'],
      };
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentCompaction: vi.fn(async () => mockResult),
      } as any);

      const tool = createMemoriCompactionTool(deps);
      const result = await tool.execute('call-1', { projectId: 'proj-1' });

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe('text');
      expect(JSON.parse(result.content[0].text)).toEqual(mockResult);
      expect(result.details).toBeNull();
    });

    it('should return error JSON when agentCompaction returns null', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentCompaction: vi.fn(async () => null),
      } as any);

      const tool = createMemoriCompactionTool(deps);
      const result = await tool.execute('call-1', { projectId: 'proj-1' });

      expect(JSON.parse(result.content[0].text)).toEqual({ error: 'Compaction failed' });
    });

    it('should return error JSON and warn when projectId is missing', async () => {
      deps.config.projectId = '';
      const tool = createMemoriCompactionTool(deps);

      const result = await tool.execute('call-1', {});

      expect(JSON.parse(result.content[0].text)).toEqual({
        error: 'projectId is required but was not configured',
      });
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_compaction rejected')
      );
    });

    it('should log params before executing', async () => {
      const tool = createMemoriCompactionTool(deps);

      await tool.execute('call-1', { projectId: 'proj-1' });

      expect(deps.logger.info).toHaveBeenCalledWith(
        expect.stringContaining('memori_compaction params')
      );
    });

    it('should return quota error and warn when QuotaExceededError is thrown', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentCompaction: vi.fn(async () => {
          throw new QuotaExceededError();
        }),
      } as any);

      const tool = createMemoriCompactionTool(deps);
      const result = await tool.execute('call-1', { projectId: 'proj-1' });

      const parsed = JSON.parse(result.content[0].text);
      expect(parsed.error).toContain('Quota exceeded');
      expect(parsed.error).toContain('100');
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_compaction quota exceeded')
      );
    });

    it('should return generic error and warn on unexpected errors', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentCompaction: vi.fn(async () => {
          throw new Error('Network failure');
        }),
      } as any);

      const tool = createMemoriCompactionTool(deps);
      const result = await tool.execute('call-1', { projectId: 'proj-1' });

      expect(JSON.parse(result.content[0].text)).toEqual({ error: 'Compaction failed' });
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_compaction failed')
      );
    });
  });
});
