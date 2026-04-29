import { describe, it, expect, vi, beforeEach } from 'vitest';
import { OpenClawIntegration } from '../../src/integrations/openclaw.js';
import { MemoriCore } from '../../src/types/integrations.js';
import { IntegrationRequest } from '../../src/types/integrations.js';

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
      project: {
        id: null,
        set: vi.fn().mockReturnThis(),
      },
      defaultApi: { post: vi.fn().mockResolvedValue(undefined) },
      collectorApi: { post: vi.fn().mockResolvedValue(undefined) },
    } as unknown as MemoriCore;

    openclaw = new OpenClawIntegration(mockCore);
  });

  describe('scope()', () => {
    it('should set the session id and return instance for chaining', () => {
      const result = openclaw.scope('my-session', 'my-project');

      expect(mockCore.session.set).toHaveBeenCalledWith('my-session');
      expect(result).toBe(openclaw);
    });

    it('should set the project id', () => {
      openclaw.scope('my-session', 'my-project');

      expect(mockCore.project.set).toHaveBeenCalledWith('my-project');
    });
  });

  describe('attribution()', () => {
    it('should update entityId and return instance for chaining', () => {
      const result = openclaw.attribution('user-123');

      expect(mockCore.config.entityId).toBe('user-123');
      expect(mockCore.config.processId).toBeUndefined();
      expect(result).toBe(openclaw);
    });

    it('should update both entityId and processId', () => {
      openclaw.attribution('user-123', 'openclaw-agent');

      expect(mockCore.config.entityId).toBe('user-123');
      expect(mockCore.config.processId).toBe('openclaw-agent');
    });
  });

  describe('augmentation()', () => {
    it('should delegate to executeAgentAugmentation', async () => {
      const spy = vi
        .spyOn(openclaw as any, 'executeAgentAugmentation')
        .mockResolvedValue(undefined);

      const req: IntegrationRequest = {
        userMessage: { role: 'user', content: 'user says hi', type: 'text' },
        agentResponse: { role: 'assistant', content: 'bot says hello', type: 'text' },
      };
      await openclaw.augmentation(req);

      expect(spy).toHaveBeenCalledWith(req);
    });
  });

  describe('recall()', () => {
    it('should delegate to executeRecall and return the result', async () => {
      const mockMemoryContext = '<memori_context>context data</memori_context>';
      const spy = vi.spyOn(openclaw as any, 'executeRecall').mockResolvedValue(mockMemoryContext);

      const result = await openclaw.recall('prompt text');

      expect(spy).toHaveBeenCalledWith('prompt text');
      expect(result).toBe(mockMemoryContext);
    });
  });

  describe('agentFeedback()', () => {
    it('should delegate to executeAgentFeedback with the provided content', async () => {
      const spy = vi.spyOn(openclaw as any, 'executeAgentFeedback').mockResolvedValue(undefined);

      await openclaw.agentFeedback('this is great');

      expect(spy).toHaveBeenCalledWith('this is great');
    });
  });
});
