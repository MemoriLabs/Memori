import { describe, it, expect, vi, beforeEach } from 'vitest';
import { initializeMemoriClient } from '../../src/utils/memori-client.js';
import type { ExtractedContext } from '../../src/utils/context.js';

vi.mock('@memorilabs/memori', () => {
  const mockOpenClawIntegration = {
    setAttribution: vi.fn(),
    setSession: vi.fn(),
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
      };

      initializeMemoriClient(apiKey, context);

      expect(Memori).toHaveBeenCalled();
      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      expect(memoriInstance.config.apiKey).toBe('test-api-key-123');
    });

    it('should integrate with OpenClawIntegration', async () => {
      const { Memori } = await import('@memorilabs/memori');
      const { OpenClawIntegration } = await import('@memorilabs/memori/integrations');

      const apiKey = 'test-api-key';
      const context: ExtractedContext = {
        entityId: 'entity-123',
        sessionId: 'session-456',
        provider: 'provider-789',
      };

      initializeMemoriClient(apiKey, context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      expect(memoriInstance.integrate).toHaveBeenCalledWith(OpenClawIntegration);
    });

    it('should set attribution with entityId and provider', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const apiKey = 'test-api-key';
      const context: ExtractedContext = {
        entityId: 'user-abc',
        sessionId: 'session-xyz',
        provider: 'openai',
      };

      initializeMemoriClient(apiKey, context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      const integration = memoriInstance.integrate.mock.results[0].value;
      expect(integration.setAttribution).toHaveBeenCalledWith('user-abc', 'openai');
    });

    it('should set session with sessionId', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const apiKey = 'test-api-key';
      const context: ExtractedContext = {
        entityId: 'entity-123',
        sessionId: 'my-session-id',
        provider: 'anthropic',
      };

      initializeMemoriClient(apiKey, context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      const integration = memoriInstance.integrate.mock.results[0].value;
      expect(integration.setSession).toHaveBeenCalledWith('my-session-id');
    });

    it('should return the OpenClawIntegration instance', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const apiKey = 'test-api-key';
      const context: ExtractedContext = {
        entityId: 'entity-123',
        sessionId: 'session-456',
        provider: 'provider-789',
      };

      const result = initializeMemoriClient(apiKey, context);

      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      const expectedIntegration = memoriInstance.integrate.mock.results[0].value;
      expect(result).toBe(expectedIntegration);
    });

    it('should configure client with correct sequence', async () => {
      const { Memori } = await import('@memorilabs/memori');

      const apiKey = 'secret-key';
      const context: ExtractedContext = {
        entityId: 'user-999',
        sessionId: 'sess-888',
        provider: 'google',
      };

      const result = initializeMemoriClient(apiKey, context);

      expect(Memori).toHaveBeenCalled();
      const memoriInstance = vi.mocked(Memori).mock.results[0].value;
      expect(memoriInstance.config.apiKey).toBe('secret-key');
      expect(memoriInstance.integrate).toHaveBeenCalled();
      expect(result.setAttribution).toHaveBeenCalledWith('user-999', 'google');
      expect(result.setSession).toHaveBeenCalledWith('sess-888');
    });
  });
});
