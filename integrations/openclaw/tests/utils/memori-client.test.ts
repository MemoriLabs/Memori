import { describe, it, expect, vi, beforeEach } from 'vitest';
import { initializeMemoriClient } from '../../src/utils/memori-client.js';
import type { ExtractedContext } from '../../src/utils/context.js';

vi.mock('@memorilabs/memori', () => {
  const mockOpenClawIntegration = {
    scope: vi.fn().mockReturnThis(),
    attribution: vi.fn().mockReturnThis(),
  };

  const mockMemori = {
    config: {},
    integrate: vi.fn(() => mockOpenClawIntegration),
  };

  return {
    Memori: vi.fn(function () {
      return mockMemori;
    }),
  };
});

vi.mock('@memorilabs/memori/integrations', () => ({
  OpenClawIntegration: class MockOpenClawIntegration {},
}));

describe('utils/memori-client', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('initializeMemoriClient', () => {
    it('should create Memori instance with API key', async () => {
      const { Memori } = await import('@memorilabs/memori');
      const apiKey = 'test-api-key-123';
      const context: ExtractedContext = {
        entityId: 'entity-456',
        sessionId: 'session-789',
        provider: 'test-provider',
        projectId: 'proj-123',
      };

      initializeMemoriClient(apiKey, context);

      expect(Memori).toHaveBeenCalled();
      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      expect(memoriInstance.config.apiKey).toBe('test-api-key-123');
    });

    it('should integrate with OpenClawIntegration', async () => {
      const { Memori } = await import('@memorilabs/memori');
      const { OpenClawIntegration } = await import('@memorilabs/memori/integrations');

      const context: ExtractedContext = {
        entityId: 'entity-123',
        sessionId: 'session-456',
        provider: 'provider-789',
        projectId: 'proj-456',
      };

      initializeMemoriClient('test-api-key', context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      expect(memoriInstance.integrate).toHaveBeenCalledWith(OpenClawIntegration);
    });

    it('should call scope with sessionId and projectId', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const context: ExtractedContext = {
        entityId: 'user-abc',
        sessionId: 'session-xyz',
        provider: 'openai',
        projectId: 'proj-xyz',
      };

      initializeMemoriClient('test-api-key', context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      const integration = memoriInstance.integrate.mock.results[0].value;
      expect(integration.scope).toHaveBeenCalledWith('session-xyz', 'proj-xyz');
    });

    it('should call attribution with entityId and provider', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const context: ExtractedContext = {
        entityId: 'user-abc',
        sessionId: 'session-xyz',
        provider: 'openai',
        projectId: 'proj-xyz',
      };

      initializeMemoriClient('test-api-key', context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      const integration = memoriInstance.integrate.mock.results[0].value;
      expect(integration.attribution).toHaveBeenCalledWith('user-abc', 'openai');
    });

    it('should return the OpenClawIntegration instance', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const context: ExtractedContext = {
        entityId: 'entity-123',
        sessionId: 'session-456',
        provider: 'provider-789',
        projectId: 'proj-789',
      };

      const result = initializeMemoriClient('test-api-key', context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      const expectedIntegration = memoriInstance.integrate.mock.results[0].value;
      expect(result).toBe(expectedIntegration);
    });

    it('should configure client with correct sequence', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const context: ExtractedContext = {
        entityId: 'user-999',
        sessionId: 'sess-888',
        provider: 'google',
        projectId: 'proj-888',
      };

      const result = initializeMemoriClient('secret-key', context);

      expect(Memori).toHaveBeenCalled();
      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      expect(memoriInstance.config.apiKey).toBe('secret-key');
      expect(memoriInstance.integrate).toHaveBeenCalled();
      expect(result.scope).toHaveBeenCalledWith('sess-888', 'proj-888');
      expect(result.attribution).toHaveBeenCalledWith('user-999', 'google');
    });
  });
});
