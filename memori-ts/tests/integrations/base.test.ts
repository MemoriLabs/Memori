import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { BaseIntegration } from '../../src/integrations/base.js';
import { MemoriCore } from '../../src/types/integrations.js';
import { LLMRequest } from '@memorilabs/axon';

// Create a concrete implementation to test the protected methods of the abstract class
class TestIntegration extends BaseIntegration {
  public testCapture(userMessage: string, agentResponse: string) {
    return this.executeCapture(userMessage, agentResponse);
  }
  public testRecall(userMessage: string) {
    return this.executeRecall(userMessage);
  }
}

describe('BaseIntegration', () => {
  let mockCore: MemoriCore;
  let integration: TestIntegration;
  let consoleWarnSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    mockCore = {
      recall: { handleRecall: vi.fn() },
      persistence: { handlePersistence: vi.fn() },
      augmentation: { handleAugmentation: vi.fn() },
      config: { entityId: 'test-user', processId: 'test-process' },
      session: { id: 'test-session-id' },
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

      await integration.testCapture('user msg', 'ai msg');

      expect(mockCore.persistence.handlePersistence).not.toHaveBeenCalled();
      expect(mockCore.augmentation.handleAugmentation).not.toHaveBeenCalled();
    });

    it('should format requests and invoke persistence and augmentation engines', async () => {
      await integration.testCapture('hello bot', 'hello human');

      const expectedReq = expect.objectContaining({
        messages: [{ role: 'user', content: 'hello bot' }],
      });
      const expectedRes = expect.objectContaining({
        content: 'hello human',
      });
      const expectedCtx = expect.objectContaining({
        traceId: expect.stringContaining('integration-trace-'),
        metadata: {},
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

      // Should not throw
      await expect(integration.testCapture('msg', 'resp')).resolves.toBeUndefined();

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
          metadata: {},
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
});
