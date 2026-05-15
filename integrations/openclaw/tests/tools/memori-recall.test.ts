import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createMemoriRecallTool } from '../../src/tools/memori-recall.js';
import type { ToolDeps } from '../../src/tools/types.js';

vi.mock('../../src/utils/memori-client.js', () => ({
  createRecallClient: vi.fn(() => ({
    agentRecall: vi.fn(async () => ({ memories: [] })),
  })),
}));

describe('tools/memori-recall', () => {
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
      const tool = createMemoriRecallTool(deps);
      expect(tool.name).toBe('memori_recall');
      expect(tool.label).toBe('Recall Memory');
    });

    it('should define expected parameter properties', () => {
      const tool = createMemoriRecallTool(deps);
      const props = tool.parameters.properties;
      expect(props).toHaveProperty('dateStart');
      expect(props).toHaveProperty('dateEnd');
      expect(props).toHaveProperty('projectId');
      expect(props).toHaveProperty('sessionId');
      expect(props).toHaveProperty('signal');
      expect(props).toHaveProperty('source');
    });
  });

  describe('execute', () => {
    it('should call createRecallClient with apiKey and entityId', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallTool(deps);

      await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-12-31',
        projectId: 'proj-1',
        sessionId: 'sess-1',
        source: 'fact',
        signal: 'verification',
      });

      expect(createRecallClient).toHaveBeenCalledWith('test-key', 'test-entity');
    });

    it('should use config.projectId as default when not supplied', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallTool(deps);

      await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-12-31',
        sessionId: 'sess-1',
        source: 'fact',
        signal: 'verification',
      });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecall).toHaveBeenCalledWith(
        expect.objectContaining({ projectId: 'default-project' })
      );
    });

    it('should allow LLM-provided projectId to override config default', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallTool(deps);

      await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-12-31',
        projectId: 'override-project',
        sessionId: 'sess-1',
        source: 'fact',
        signal: 'verification',
      });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecall).toHaveBeenCalledWith(
        expect.objectContaining({ projectId: 'override-project' })
      );
    });

    it('should pass all optional params through to agentRecall', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallTool(deps);

      await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-12-31',
        projectId: 'proj-1',
        sessionId: 'sess-1',
        source: 'fact',
        signal: 'verification',
      });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecall).toHaveBeenCalledWith(
        expect.objectContaining({
          dateStart: '2024-01-01',
          dateEnd: '2024-12-31',
          projectId: 'proj-1',
          sessionId: 'sess-1',
          source: 'fact',
          signal: 'verification',
        })
      );
    });

    it('should return JSON-stringified result on success', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const mockResult = { memories: [{ id: '1', text: 'Remember this' }] };
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentRecall: vi.fn(async () => mockResult),
      } as any);

      const tool = createMemoriRecallTool(deps);
      const result = await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-12-31',
        projectId: 'proj-1',
        sessionId: 'sess-1',
        source: 'fact',
        signal: 'verification',
      });

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe('text');
      expect(JSON.parse(result.content[0].text)).toEqual(mockResult);
      expect(result.details).toBeNull();
    });

    it('should return error JSON and warn when agentRecall throws', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentRecall: vi.fn(async () => {
          throw new Error('Network error');
        }),
      } as any);

      const tool = createMemoriRecallTool(deps);
      const result = await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-12-31',
        projectId: 'proj-1',
        sessionId: 'sess-1',
        source: 'fact',
        signal: 'verification',
      });

      expect(JSON.parse(result.content[0].text)).toEqual({ error: 'Recall failed' });
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_recall failed')
      );
    });

    it('should reject sessionId when projectId resolves to empty', async () => {
      deps.config.projectId = '';
      const tool = createMemoriRecallTool(deps);

      const result = await tool.execute('call-1', {
        projectId: '',
        sessionId: 'sess-1',
      });

      expect(JSON.parse(result.content[0].text)).toEqual({
        error: 'sessionId cannot be provided without projectId',
      });
    });

    it('should allow sessionId when projectId is present', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriRecallTool(deps);

      const result = await tool.execute('call-1', {
        projectId: 'proj-1',
        sessionId: 'sess-1',
      });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentRecall).toHaveBeenCalled();
      expect(JSON.parse(result.content[0].text)).not.toHaveProperty('error');
    });

    it('should log params before executing', async () => {
      const tool = createMemoriRecallTool(deps);

      await tool.execute('call-1', {
        dateStart: '2024-01-01',
        dateEnd: '2024-12-31',
        projectId: 'proj-1',
        sessionId: 'sess-1',
        source: 'fact',
        signal: 'verification',
      });

      expect(deps.logger.info).toHaveBeenCalledWith(
        expect.stringContaining('memori_recall params')
      );
    });

    describe('source/signal validation', () => {
      it('should reject source provided without signal', async () => {
        const tool = createMemoriRecallTool(deps);

        const result = await tool.execute('call-1', { source: 'fact' });

        expect(JSON.parse(result.content[0].text)).toEqual({
          error: 'source and signal must be provided together or both omitted',
        });
        expect(deps.logger.warn).toHaveBeenCalledWith(
          expect.stringContaining('memori_recall rejected')
        );
      });

      it('should reject signal provided without source', async () => {
        const tool = createMemoriRecallTool(deps);

        const result = await tool.execute('call-1', { signal: 'verification' });

        expect(JSON.parse(result.content[0].text)).toEqual({
          error: 'source and signal must be provided together or both omitted',
        });
        expect(deps.logger.warn).toHaveBeenCalledWith(
          expect.stringContaining('memori_recall rejected')
        );
      });

      it('should reject an invalid (source, signal) pair', async () => {
        const tool = createMemoriRecallTool(deps);

        const result = await tool.execute('call-1', { source: 'fact', signal: 'commit' });

        const parsed = JSON.parse(result.content[0].text);
        expect(parsed.error).toMatch(/Invalid \(source, signal\) pair/);
        expect(parsed.error).toContain('fact');
        expect(parsed.error).toContain('commit');
        expect(parsed.error).toContain('verification');
        expect(deps.logger.warn).toHaveBeenCalledWith(
          expect.stringContaining('memori_recall rejected')
        );
      });

      it('should succeed when both source and signal are omitted', async () => {
        const { createRecallClient } = await import('../../src/utils/memori-client.js');
        const tool = createMemoriRecallTool(deps);

        const result = await tool.execute('call-1', { projectId: 'proj-1' });

        const client = vi.mocked(createRecallClient).mock.results[0].value;
        expect(client.agentRecall).toHaveBeenCalled();
        expect(JSON.parse(result.content[0].text)).not.toHaveProperty('error');
      });

      it.each([
        ['constraint', 'discovery'],
        ['decision', 'commit'],
        ['fact', 'verification'],
        ['execution', 'failure'],
        ['instruction', 'discovery'],
        ['insight', 'inference'],
        ['status', 'update'],
        ['strategy', 'pattern'],
        ['task', 'result'],
      ])('should accept valid pair (%s, %s)', async (source, signal) => {
        const { createRecallClient } = await import('../../src/utils/memori-client.js');
        const tool = createMemoriRecallTool(deps);

        const result = await tool.execute('call-1', { source, signal });

        const client = vi.mocked(createRecallClient).mock.results[0].value;
        expect(client.agentRecall).toHaveBeenCalledWith(
          expect.objectContaining({ source, signal })
        );
        expect(JSON.parse(result.content[0].text)).not.toHaveProperty('error');
      });
    });
  });
});
