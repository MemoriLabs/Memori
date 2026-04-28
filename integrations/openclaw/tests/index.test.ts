import { describe, it, expect, vi, beforeEach } from 'vitest';
import memoriPlugin from '../src/index.js';
import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';

vi.mock('../src/utils/skills-loader.js', () => ({
  loadSkillsContent: vi.fn(() => 'mock skills content'),
}));

vi.mock('../src/handlers/augmentation.js', () => ({
  handleAugmentation: vi.fn(),
}));

describe('plugin index', () => {
  let mockApi: OpenClawPluginApi;
  let mockOn: ReturnType<typeof vi.fn>;
  let mockRegisterTool: ReturnType<typeof vi.fn>;

  beforeEach(() => {
    vi.clearAllMocks();

    mockOn = vi.fn();
    mockRegisterTool = vi.fn();

    mockApi = {
      pluginConfig: {
        apiKey: 'test-api-key',
        entityId: 'test-entity-id',
      },
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
      },
      on: mockOn,
      registerTool: mockRegisterTool,
      resolvePath: vi.fn(
        (relativePath: string) => `/tmp/memori-openclaw-test-missing/${relativePath}`
      ),
      registerCli: vi.fn(),
    } as unknown as OpenClawPluginApi;
  });

  describe('plugin metadata', () => {
    it('should have correct id', () => {
      expect(memoriPlugin.id).toBe('openclaw-memori');
    });

    it('should have correct name', () => {
      expect(memoriPlugin.name).toBe('Memori System');
    });

    it('should have description', () => {
      expect(memoriPlugin.description).toBe('Hosted memory backend');
    });
  });

  describe('register', () => {
    it('should register hooks when config is valid', () => {
      memoriPlugin.register(mockApi);

      expect(mockOn).toHaveBeenCalledWith('before_prompt_build', expect.any(Function));
      expect(mockOn).toHaveBeenCalledWith('agent_end', expect.any(Function));
      expect(mockOn).toHaveBeenCalledTimes(2);
    });

    it('should not register when apiKey is missing', () => {
      mockApi.pluginConfig = {
        entityId: 'test-entity-id',
      };

      memoriPlugin.register(mockApi);

      expect(mockApi.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Missing apiKey or entityId')
      );
      expect(mockOn.mock.calls).toHaveLength(0);
    });

    it('should not register when entityId is missing', () => {
      mockApi.pluginConfig = {
        apiKey: 'test-api-key',
      };

      memoriPlugin.register(mockApi);

      expect(mockApi.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Missing apiKey or entityId')
      );
      expect(mockOn.mock.calls).toHaveLength(0);
    });

    it('should not register when both apiKey and entityId are missing', () => {
      mockApi.pluginConfig = {};

      memoriPlugin.register(mockApi);

      expect(mockApi.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Missing apiKey or entityId')
      );
      expect(mockOn.mock.calls).toHaveLength(0);
    });

    it('should not register when pluginConfig is undefined', () => {
      mockApi.pluginConfig = undefined;

      memoriPlugin.register(mockApi);

      expect(mockApi.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Missing apiKey or entityId')
      );
      expect(mockOn.mock.calls).toHaveLength(0);
    });

    it('should not register when apiKey is empty string', () => {
      mockApi.pluginConfig = {
        apiKey: '',
        entityId: 'test-entity-id',
      };

      memoriPlugin.register(mockApi);

      expect(mockApi.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Missing apiKey or entityId')
      );
      expect(mockOn.mock.calls).toHaveLength(0);
    });

    it('should not register when entityId is empty string', () => {
      mockApi.pluginConfig = {
        apiKey: 'test-api-key',
        entityId: '',
      };

      memoriPlugin.register(mockApi);

      expect(mockApi.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('Missing apiKey or entityId')
      );
      expect(mockOn.mock.calls).toHaveLength(0);
    });
  });

  describe('hook handlers', () => {
    it('should inject skills content via before_prompt_build handler', () => {
      memoriPlugin.register(mockApi);

      const handler = mockOn.mock.calls.find((call) => call[0] === 'before_prompt_build')?.[1];

      expect(handler).toBeDefined();
      expect(handler?.()).toEqual({ appendSystemContext: 'mock skills content' });
    });

    it('should call handleAugmentation for agent_end event', async () => {
      const { handleAugmentation } = await import('../src/handlers/augmentation.js');

      memoriPlugin.register(mockApi);

      const agentEndHandler = mockOn.mock.calls.find((call) => call[0] === 'agent_end')?.[1];

      expect(agentEndHandler).toBeDefined();

      const mockEvent = { success: true, messages: [] } as any;
      const mockCtx = { sessionKey: 'session-123' } as any;

      await agentEndHandler?.(mockEvent, mockCtx);

      expect(handleAugmentation).toHaveBeenCalledWith(
        mockEvent,
        mockCtx,
        { apiKey: 'test-api-key', entityId: 'test-entity-id' },
        expect.any(Object)
      );
    });
  });

  describe('configuration handling', () => {
    it('should extract apiKey from pluginConfig', () => {
      mockApi.pluginConfig = {
        apiKey: 'custom-key-123',
        entityId: 'entity-456',
      };

      memoriPlugin.register(mockApi);

      expect(mockOn.mock.calls.length).toBeGreaterThan(0);
    });

    it('should handle additional config properties gracefully', () => {
      mockApi.pluginConfig = {
        apiKey: 'test-api-key',
        entityId: 'test-entity-id',
        extraProperty: 'should be ignored',
        anotherExtra: 12345,
      };

      memoriPlugin.register(mockApi);

      expect(mockOn.mock.calls).toHaveLength(2);
    });
  });

  describe('logger creation', () => {
    it('should create MemoriLogger with api', () => {
      memoriPlugin.register(mockApi);

      expect(mockApi.logger.info).toHaveBeenCalledWith(expect.stringContaining('[Memori]'));
    });
  });
});
