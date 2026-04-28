import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createMemoriRecallSummaryTool } from '../../src/tools/memori-recall-summary.js';
import type { ToolDeps } from '../../src/tools/types.js';

vi.mock('../../src/utils/memori-client.js', () => ({
  createRecallClient: vi.fn(() => ({
    agentRecallSummary: vi.fn(async () => ({ summaries: [] })),
  })),
}));

describe('tools/memori-recall-summary', () => {
  let deps: ToolDeps;

  beforeEach(async () => {
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
      const tool = createMemoriRecallSummaryTool(deps);
      expect(tool.name).toBe('memori_recall_summary');
      expect(tool.label).toBe('Recall Memory Summary');
    });

    it('should define expected parameter properties', () => {
      const tool = createMemoriRecallSummaryTool(deps);
      const props = tool.parameters.properties;
      expect(props).toHaveProperty('dateStart');
      expect(props).toHaveProperty('dateEnd');
      expect(props).toHaveProperty('projectId');
      expect(props).toHaveProperty('sessionId');
    });
  });

  describe('execute', () => {
    it('should call createRecallClient with apiKey and entityId', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallSummaryTool(deps);

      await tool.execute('call-1', {});

      expect(createRecallClient).toHaveBeenCalledWith('test-key', 'test-entity');
    });

    it('should use config.projectId as default when not supplied', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallSummaryTool(deps);

      await tool.execute('call-1', {});

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecallSummary).toHaveBeenCalledWith(
        expect.objectContaining({ projectId: 'default-project' })
      );
    });

    it('should allow LLM-provided projectId to override config default', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallSummaryTool(deps);

      await tool.execute('call-1', { projectId: 'override-project' });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecallSummary).toHaveBeenCalledWith(
        expect.objectContaining({ projectId: 'override-project' })
      );
    });

    it('should pass date filters through to agentRecallSummary', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallSummaryTool(deps);

      await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-06-30',
        projectId: 'proj-1',
        sessionId: 'sess-1',
      });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecallSummary).toHaveBeenCalledWith(
        expect.objectContaining({
          dateStart: '2024-01-01',
          dateEnd: '2024-06-30',
          projectId: 'proj-1',
          sessionId: 'sess-1',
        })
      );
    });

    it('should return JSON-stringified result on success', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const mockResult = { summaries: [{ id: 's1', text: 'Summary text' }] };
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentRecallSummary: vi.fn(async () => mockResult),
      } as any);

      const tool = createMemoriRecallSummaryTool(deps);
      const result = await tool.execute('call-1', {});

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe('text');
      expect(JSON.parse(result.content[0].text)).toEqual(mockResult);
      expect(result.details).toBeNull();
    });

    it('should return error JSON and warn when agentRecallSummary throws', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentRecallSummary: vi.fn(async () => {
          throw new Error('Server error');
        }),
      } as any);

      const tool = createMemoriRecallSummaryTool(deps);
      const result = await tool.execute('call-1', {});

      expect(JSON.parse(result.content[0].text)).toEqual({ error: 'Recall summary failed' });
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_recall_summary failed')
      );
    });

    it('should reject sessionId when projectId resolves to empty', async () => {
      deps.config.projectId = '';
      const tool = createMemoriRecallSummaryTool(deps);

      const result = await tool.execute('call-1', { projectId: '', sessionId: 'sess-1' });

      expect(JSON.parse(result.content[0].text)).toEqual({
        error: 'sessionId cannot be provided without projectId',
      });
    });

    it('should allow sessionId when projectId is present', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallSummaryTool(deps);

      const result = await tool.execute('call-1', { projectId: 'proj-1', sessionId: 'sess-1' });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecallSummary).toHaveBeenCalled();
      expect(JSON.parse(result.content[0].text)).not.toHaveProperty('error');
    });

    it('should log params before executing', async () => {
      const tool = createMemoriRecallSummaryTool(deps);

      await tool.execute('call-1', { projectId: 'proj-1' });

      expect(deps.logger.info).toHaveBeenCalledWith(
        expect.stringContaining('memori_recall_summary params')
      );
    });
  });
});
