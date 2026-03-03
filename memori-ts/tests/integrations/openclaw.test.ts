import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenClawIntegration } from '../../src/integrations/openclaw.js';
import { MemoriCore } from '../../src/types/integrations.js';

describe('OpenClawIntegration', () => {
  let mockCore: MemoriCore;
  let openclaw: OpenClawIntegration;

  beforeEach(() => {
    mockCore = {
      recall: {} as any,
      persistence: {} as any,
      augmentation: {} as any,
      config: { entityId: undefined, processId: undefined },
      session: {
        id: 'default-session-id',
        set: vi.fn().mockReturnThis(),
      },
    } as unknown as MemoriCore;

    openclaw = new OpenClawIntegration(mockCore);
  });

  describe('setAttribution()', () => {
    it('should update entityId and return instance for chaining', () => {
      const result = openclaw.setAttribution('user-123');

      expect(mockCore.config.entityId).toBe('user-123');
      expect(mockCore.config.processId).toBeUndefined();
      expect(result).toBe(openclaw); // Chainable
    });

    it('should update both entityId and processId', () => {
      openclaw.setAttribution('user-123', 'openclaw-agent');

      expect(mockCore.config.entityId).toBe('user-123');
      expect(mockCore.config.processId).toBe('openclaw-agent');
    });
  });

  describe('setSession()', () => {
    it('should delegate to core session manager and return instance for chaining', () => {
      const result = openclaw.setSession('custom-session-uuid');

      expect(mockCore.session.set).toHaveBeenCalledWith('custom-session-uuid');
      expect(result).toBe(openclaw); // Chainable
    });
  });

  describe('capture()', () => {
    it('should delegate to the inherited executeCapture method', async () => {
      // Spy on the protected method inherited from BaseIntegration
      const executeCaptureSpy = vi
        .spyOn(openclaw as any, 'executeCapture')
        .mockResolvedValue(undefined);

      await openclaw.capture('user says hi', 'bot says hello');

      expect(executeCaptureSpy).toHaveBeenCalledWith('user says hi', 'bot says hello');
    });
  });

  describe('recall()', () => {
    it('should delegate to the inherited executeRecall method and return the result', async () => {
      const mockMemoryContext = '<memori_context>context data</memori_context>';

      // Spy on the protected method inherited from BaseIntegration
      const executeRecallSpy = vi
        .spyOn(openclaw as any, 'executeRecall')
        .mockResolvedValue(mockMemoryContext);

      const result = await openclaw.recall('prompt text');

      expect(executeRecallSpy).toHaveBeenCalledWith('prompt text');
      expect(result).toBe(mockMemoryContext);
    });
  });
});
