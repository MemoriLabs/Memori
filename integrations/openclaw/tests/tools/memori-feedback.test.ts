import { describe, it, expect, vi, beforeEach } from 'vitest';
import { createMemoriFeedbackTool } from '../../src/tools/memori-feedback.js';
import type { ToolDeps } from '../../src/tools/types.js';

vi.mock('../../src/utils/memori-client.js', () => ({
  createRecallClient: vi.fn(() => ({
    agentFeedback: vi.fn(async () => undefined),
  })),
}));

describe('tools/memori-feedback', () => {
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
      const tool = createMemoriFeedbackTool(deps);
      expect(tool.name).toBe('memori_feedback');
      expect(tool.label).toBe('Memori Feedback');
    });

    it('should require the content parameter', () => {
      const tool = createMemoriFeedbackTool(deps);
      expect(tool.parameters.required).toContain('content');
    });

    it('should define the content parameter property', () => {
      const tool = createMemoriFeedbackTool(deps);
      expect(tool.parameters.properties).toHaveProperty('content');
    });
  });

  describe('execute', () => {
    it('should call createRecallClient with apiKey and entityId', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriFeedbackTool(deps);

      await tool.execute('call-1', { content: 'great product!' });

      expect(createRecallClient).toHaveBeenCalledWith('test-key', 'test-entity');
    });

    it('should call agentFeedback with the provided content', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      const tool = createMemoriFeedbackTool(deps);

      await tool.execute('call-1', { content: 'please add dark mode' });

      const client = vi.mocked(createRecallClient).mock.results[0].value;
      expect(client.agentFeedback).toHaveBeenCalledWith('please add dark mode');
    });

    it('should return success JSON on success', async () => {
      const tool = createMemoriFeedbackTool(deps);
      const result = await tool.execute('call-1', { content: 'love it' });

      expect(result.content).toHaveLength(1);
      expect(result.content[0].type).toBe('text');
      expect(JSON.parse(result.content[0].text)).toEqual({
        success: true,
        message: 'Feedback sent successfully.',
      });
      expect(result.details).toBeNull();
    });

    it('should return error JSON and warn when agentFeedback throws', async () => {
      const { createRecallClient } = await import('../../src/utils/memori-client.js');
      vi.mocked(createRecallClient).mockReturnValueOnce({
        agentFeedback: vi.fn(async () => {
          throw new Error('Network error');
        }),
      } as any);

      const tool = createMemoriFeedbackTool(deps);
      const result = await tool.execute('call-1', { content: 'bad feedback' });

      expect(JSON.parse(result.content[0].text)).toEqual({ error: 'Feedback failed to send.' });
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_feedback failed')
      );
    });

    it('should log the feedback content before sending', async () => {
      const tool = createMemoriFeedbackTool(deps);

      await tool.execute('call-1', { content: 'log this' });

      expect(deps.logger.info).toHaveBeenCalledWith(
        expect.stringContaining('memori_feedback sending')
      );
    });
  });
});
