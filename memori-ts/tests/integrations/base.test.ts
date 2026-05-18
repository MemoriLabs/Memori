import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { BaseIntegration } from '../../src/integrations/base.js';
import { IntegrationRequest, MemoriCore } from '../../src/types/integrations.js';
import { LLMRequest } from '@memorilabs/axon';

// Create a concrete implementation to test the protected methods of the abstract class
class TestIntegration extends BaseIntegration {
  public testCapture(req: IntegrationRequest) {
    return this.executeAugmentation(req);
  }
  public testRecall(userMessage: string) {
    return this.executeRecall(userMessage);
  }
  public testAgentFeedback(content: string) {
    return this.executeAgentFeedback(content);
  }
  public testAgentAugmentation(req: IntegrationRequest) {
    return this.executeAgentAugmentation(req);
  }
  public testAgentRecall(params?: Parameters<typeof this.executeAgentRecall>[0]) {
    return this.executeAgentRecall(params);
  }
  public testAgentRecallSummary(params?: Parameters<typeof this.executeAgentRecallSummary>[0]) {
    return this.executeAgentRecallSummary(params);
  }
  public testAgentCompaction(params: Parameters<typeof this.executeAgentCompaction>[0]) {
    return this.executeAgentCompaction(params);
  }
}

describe('BaseIntegration', () => {
  let mockCore: MemoriCore;
  let integration: TestIntegration;
  let consoleWarnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    mockCore = {
      recall: {
        handleRecall: vi.fn(),
        agentRecall: vi.fn(),
        agentRecallSummary: vi.fn(),
        agentCompaction: vi.fn(),
      },
      persistence: { handlePersistence: vi.fn() },
      augmentation: { handleAugmentation: vi.fn(), handleAgentAugmentation: vi.fn() },
      config: { entityId: 'test-user', processId: 'test-process' },
      session: { id: 'test-session-id' },
      project: { id: 'test-project-id', set: vi.fn() },
      defaultApi: { post: vi.fn().mockResolvedValue(undefined) },
      collectorApi: { post: vi.fn().mockResolvedValue(undefined) },
    } as unknown as MemoriCore;

    integration = new TestIntegration(mockCore);
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    consoleWarnSpy.mockRestore();
  });

  describe('executeCapture()', () => {
    it('should silently abort if no session ID is present', async () => {
      (mockCore.session as any).id = undefined;

      const req: IntegrationRequest = {
        userMessage: { role: 'user', content: 'user msg', type: 'text' },
        agentResponse: { role: 'assistant', content: 'ai msg', type: 'text' },
      };

      await integration.testCapture(req);

      expect(mockCore.persistence.handlePersistence).not.toHaveBeenCalled();
      expect(mockCore.augmentation.handleAugmentation).not.toHaveBeenCalled();
    });

    it('should format requests and invoke engines, properly passing metadata', async () => {
      const req: IntegrationRequest = {
        userMessage: { role: 'user', content: 'hello bot', type: 'text' },
        agentResponse: { role: 'assistant', content: 'hello human', type: 'text' },
        metadata: {
          provider: 'openclaw',
          model: 'gpt-4o',
          platform: 'openclaw',
          sdkVersion: null,
          integrationSdkVersion: '1.0.0',
        },
      };

      await integration.testCapture(req);

      const expectedReq = expect.objectContaining({
        messages: [{ role: 'user', content: 'hello bot', type: 'text' }],
        model: 'gpt-4o',
      });
      const expectedRes = expect.objectContaining({
        content: 'hello human',
        type: 'text',
      });
      const expectedCtx = expect.objectContaining({
        traceId: expect.stringContaining('integration-trace-'),
        metadata: expect.objectContaining({
          ...req.metadata,
          memoriTrace: null,
        }),
      });

      expect(mockCore.persistence.handlePersistence).toHaveBeenCalledWith(
        expectedReq,
        expectedRes,
        expectedCtx
      );
      expect(mockCore.augmentation.handleAugmentation).toHaveBeenCalledWith(
        expectedReq,
        expectedRes,
        expectedCtx
      );
    });

    it('should swallow errors and log a warning if engines fail', async () => {
      (mockCore.persistence.handlePersistence as any).mockRejectedValue(
        new Error('Persistence failed')
      );

      const req: IntegrationRequest = {
        userMessage: { role: 'user', content: 'msg', type: 'text' },
        agentResponse: { role: 'assistant', content: 'resp', type: 'text' },
      };

      // Should not throw
      await expect(integration.testCapture(req)).resolves.toBeUndefined();

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Memori Integration Capture failed:',
        expect.any(Error)
      );
    });
  });

  describe('executeRecall()', () => {
    it('should return undefined if no session ID is present', async () => {
      (mockCore.session as any).id = undefined;

      const result = await integration.testRecall('who am i?');

      expect(result).toBeUndefined();
      expect(mockCore.recall.handleRecall).not.toHaveBeenCalled();
    });

    it('should format the request, invoke the recall engine, and extract the system message', async () => {
      const mockUpdatedReq: LLMRequest = {
        messages: [
          { role: 'system', content: '<memori_context>You like apples.</memori_context>' },
          { role: 'user', content: 'what do I like?' },
        ],
      };
      (mockCore.recall.handleRecall as any).mockResolvedValue(mockUpdatedReq);

      const result = await integration.testRecall('what do I like?');

      expect(mockCore.recall.handleRecall).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [{ role: 'user', content: 'what do I like?' }],
        }),
        expect.objectContaining({
          traceId: expect.stringContaining('integration-trace-'),
          metadata: { memoriTrace: null },
        })
      );
      expect(result).toBe('<memori_context>You like apples.</memori_context>');
    });

    it('should return undefined if the recall engine does not inject a system message', async () => {
      const mockUpdatedReq: LLMRequest = {
        messages: [{ role: 'user', content: 'what do I like?' }],
      };
      (mockCore.recall.handleRecall as any).mockResolvedValue(mockUpdatedReq);

      const result = await integration.testRecall('what do I like?');

      expect(result).toBeUndefined();
    });

    it('should swallow errors, log a warning, and return undefined on failure', async () => {
      (mockCore.recall.handleRecall as any).mockRejectedValue(new Error('Recall failed'));

      const result = await integration.testRecall('query');

      expect(result).toBeUndefined();
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Memori Integration Recall failed:',
        expect.any(Error)
      );
    });
  });

  describe('executeAgentFeedback()', () => {
    it('should POST to agent/feedback with the provided content', async () => {
      await integration.testAgentFeedback('great product!');

      expect(mockCore.defaultApi.post).toHaveBeenCalledWith('agent/feedback', {
        content: 'great product!',
      });
    });

    it('should swallow errors and log a warning on failure', async () => {
      (mockCore.defaultApi.post as any).mockRejectedValue(new Error('Network error'));

      await expect(integration.testAgentFeedback('oops')).resolves.toBeUndefined();

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Memori Agent Feedback failed:',
        expect.any(Error)
      );
    });
  });

  describe('executeAgentAugmentation()', () => {
    const req: IntegrationRequest = {
      userMessage: { role: 'user', content: 'hello bot', type: 'text' },
      agentResponse: { role: 'assistant', content: 'hello human', type: 'text' },
      trace: { tools: [{ name: 'search', args: { query: 'hello' }, result: 'found' }] },
    };

    it('should silently abort if no session ID is present', async () => {
      (mockCore.session as any).id = undefined;

      await integration.testAgentAugmentation(req);

      expect(mockCore.persistence.handlePersistence).not.toHaveBeenCalled();
      expect(mockCore.augmentation.handleAgentAugmentation).not.toHaveBeenCalled();
    });

    it('should call handlePersistence and handleAgentAugmentation with correct payloads', async () => {
      await integration.testAgentAugmentation(req);

      expect(mockCore.persistence.handlePersistence).toHaveBeenCalledWith(
        expect.objectContaining({
          messages: [{ role: 'user', content: 'hello bot', type: 'text' }],
        }),
        expect.objectContaining({ content: 'hello human', type: 'text' }),
        expect.objectContaining({ traceId: expect.stringContaining('integration-trace-') })
      );
      expect(mockCore.augmentation.handleAgentAugmentation).toHaveBeenCalledWith(
        expect.anything(),
        expect.anything(),
        expect.anything(),
        req.trace,
        req.summary
      );
    });

    it('should swallow errors and log a warning if engines fail', async () => {
      (mockCore.persistence.handlePersistence as any).mockRejectedValue(
        new Error('Persistence failed')
      );

      await expect(integration.testAgentAugmentation(req)).resolves.toBeUndefined();

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Memori Integration Capture failed:',
        expect.any(Error)
      );
    });
  });

  describe('executeAgentRecall()', () => {
    it('should delegate to core.recall.agentRecall and return the result', async () => {
      const mockResult = { facts: [{ id: 1, content: 'memory' }] };
      (mockCore.recall.agentRecall as any).mockResolvedValue(mockResult);

      const result = await integration.testAgentRecall({ projectId: 'proj-1' });

      expect(mockCore.recall.agentRecall).toHaveBeenCalledWith({ projectId: 'proj-1' });
      expect(result).toEqual(mockResult);
    });

    it('should return an empty object and log a warning when agentRecall throws', async () => {
      (mockCore.recall.agentRecall as any).mockRejectedValue(new Error('API error'));

      const result = await integration.testAgentRecall({ projectId: 'proj-1' });

      expect(result).toEqual({});
      expect(consoleWarnSpy).toHaveBeenCalledWith('Memori Agent Recall failed:', expect.any(Error));
    });
  });

  describe('executeAgentRecallSummary()', () => {
    it('should delegate to core.recall.agentRecallSummary and return the result', async () => {
      const mockResult = {
        summaries: [
          { content: 'summary', date_created: '2024-01-01', entity_fact_id: 1, fact_id: 1 },
        ],
      };
      (mockCore.recall.agentRecallSummary as any).mockResolvedValue(mockResult);

      const result = await integration.testAgentRecallSummary({ projectId: 'proj-1' });

      expect(mockCore.recall.agentRecallSummary).toHaveBeenCalledWith({ projectId: 'proj-1' });
      expect(result).toEqual(mockResult);
    });

    it('should return an empty object and log a warning when agentRecallSummary throws', async () => {
      (mockCore.recall.agentRecallSummary as any).mockRejectedValue(new Error('API error'));

      const result = await integration.testAgentRecallSummary();

      expect(result).toEqual({});
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Memori Agent Recall Summary failed:',
        expect.any(Error)
      );
    });
  });

  describe('executeAgentCompaction()', () => {
    const mockCompactionResponse = {
      continuation: { last_action: 'ran tests', next_expected_action: 'open PR' },
      environment: ['CI=true'],
      messages: [{ content: 'hi', role: 'user', type: 'text' }],
      metadata: {
        date: { execution: '2024-06-01T00:00:00.000Z' },
        filter: { project: { id: 'proj-1' } },
      },
      standing_orders: ['prefer small commits'],
      state: { active_tasks: ['auth refactor'], open_loops: [], pending_results: [] },
      workspace_changes: [],
    };

    it('should delegate to core.recall.agentCompaction and return the result', async () => {
      (mockCore.recall.agentCompaction as any).mockResolvedValue(mockCompactionResponse);

      const result = await integration.testAgentCompaction({ projectId: 'proj-1' });

      expect(mockCore.recall.agentCompaction).toHaveBeenCalledWith({ projectId: 'proj-1' });
      expect(result).toEqual(mockCompactionResponse);
    });

    it('should forward sessionId and numMessages params', async () => {
      (mockCore.recall.agentCompaction as any).mockResolvedValue(mockCompactionResponse);

      await integration.testAgentCompaction({
        projectId: 'proj-1',
        sessionId: 'sess-1',
        numMessages: 10,
      });

      expect(mockCore.recall.agentCompaction).toHaveBeenCalledWith({
        projectId: 'proj-1',
        sessionId: 'sess-1',
        numMessages: 10,
      });
    });

    it('should return null and log a warning when agentCompaction throws', async () => {
      (mockCore.recall.agentCompaction as any).mockRejectedValue(new Error('API error'));

      const result = await integration.testAgentCompaction({ projectId: 'proj-1' });

      expect(result).toBeNull();
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Memori Agent Compaction failed:',
        expect.any(Error)
      );
    });
  });
});
