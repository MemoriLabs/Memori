import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleAugmentation } from '../../src/handlers/augmentation.js';
import type {
  OpenClawEvent,
  OpenClawContext,
  MemoriPluginConfig,
} from '../../src/types.js';
import type { MemoriLogger } from '../../src/utils/logger.js';
import { SDK_VERSION } from '../../src/version.js';
import { MESSAGE_CONSTANTS } from '../../src/constants.js';

vi.mock('../../src/sanitizer.js', () => ({
  cleanText: vi.fn((content) => {
    if (typeof content === 'string') return content;
    return ''; 
  }),
  isSystemMessage: vi.fn(() => false),
}));

vi.mock('../../src/utils/index.js', () => ({
  extractContext: vi.fn(() => ({
    entityId: 'test-entity',
    sessionId: 'test-session',
    provider: 'test-provider',
    projectId: 'test-project',
  })),
  initializeMemoriClient: vi.fn(() => ({
    augmentation: vi.fn(async () => {}),
  })),
}));

describe('handlers/augmentation', () => {
  let mockLogger: MemoriLogger;
  let config: MemoriPluginConfig;
  let event: OpenClawEvent;
  let ctx: OpenClawContext;

  beforeEach(() => {
    vi.clearAllMocks();

    mockLogger = {
      section: vi.fn(),
      endSection: vi.fn(),
      info: vi.fn(),
      error: vi.fn(),
    } as unknown as MemoriLogger;

    config = { apiKey: 'test-api-key', entityId: 'test-entity-id', projectId: 'test-project-id' };
    ctx = { sessionKey: 'session-123', messageProvider: 'test-provider' };
    event = {
      success: true,
      messages: [
        { role: 'user', content: 'Hello' },
        { role: 'assistant', content: "Hi" },
      ],
    };
  });

  describe('edge case coverage', () => {
    /**
     * TARGETS: Lines 51-54 & 62
     * Logic: parseToolArguments catch block and final return {}
     */
    it('should handle malformed JSON and unexpected argument types in tool calls', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      
      event.messages = [
        { role: 'user', content: 'run tool' },
        {
          role: 'assistant',
          content: [
            // Line 51-54: String that is NOT valid JSON
            { type: 'toolCall', id: '1', name: 'bad_json', arguments: '{malformed' },
            // Line 62: Unexpected type (null or non-object)
            { type: 'toolCall', id: '2', name: 'null_args', arguments: null }
          ]
        }
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      const tools = vi.mocked(client.augmentation).mock.calls[0][0].trace?.tools;
      
      // Both should fall back to empty objects {}
      expect(tools?.[0].args).toEqual({});
      expect(tools?.[1].args).toEqual({});
    });

    /**
     * TARGETS: Lines 202-203
     * Logic: Catch block for the main handler error logging
     */
    it('should log an error when context extraction fails', async () => {
      const { extractContext } = await import('../../src/utils/index.js');
      vi.mocked(extractContext).mockImplementationOnce(() => {
        throw new Error('Context resolution failed');
      });

      await handleAugmentation(event, ctx, config, mockLogger);

      // Verify the catch block (Line 202) is hit and logged
      expect(mockLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Augmentation failed: Context resolution failed')
      );
    });
  });

  describe('standard conversation (no tools)', () => {
    it('should call augmentation without a trace object', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => typeof c === 'string' ? c : 'response text');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          userMessage: 'Hello',
          agentResponse: "Hi",
        })
      );

      const payload = vi.mocked(client.augmentation).mock.calls[0][0];
      expect(payload).not.toHaveProperty('trace');
    });

    it('should include LLM metadata in request', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => typeof c === 'string' ? c : 'text');
      
      event.messages![1].provider = 'anthropic';
      event.messages![1].model = 'claude-3-5-sonnet';

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: {
            provider: 'anthropic',
            model: 'claude-3-5-sonnet',
            sdkVersion: null,
            integrationSdkVersion: SDK_VERSION,
            platform: 'openclaw',
          },
        })
      );
    });
  });

  describe('tool call extraction', () => {
    it('should extract OpenAI format tool calls (toolCall)', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : ''));

      event.messages = [
        { role: 'user', content: 'What time is it?' },
        {
          role: 'assistant',
          content: [
            {
              type: 'toolCall',
              id: 'call_1',
              name: 'get_time',
              arguments: { timezone: 'UTC' },
            },
          ],
        },
        { role: 'toolResult', toolCallId: 'call_1', content: '2024-03-21T10:00:00Z' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          trace: {
            tools: [{ name: 'get_time', args: { timezone: 'UTC' }, result: '2024-03-21T10:00:00Z' }],
          },
        })
      );
    });

    it('should generate a synthetic response for silent tool executions', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : ''));

      event.messages = [
        { role: 'user', content: 'Update state' },
        {
          role: 'assistant',
          content: [{ type: 'toolCall', id: '3', name: 'update', arguments: {} }],
        },
        { role: 'toolResult', toolCallId: '3', content: 'ok' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          agentResponse: MESSAGE_CONSTANTS.SYNTHETIC_RESPONSE,
        })
      );
    });
  });

  describe('validation and skipping', () => {
    it('should skip when event is unsuccessful', async () => {
      event.success = false;
      await handleAugmentation(event, ctx, config, mockLogger);
      expect(mockLogger.info).toHaveBeenCalledWith(expect.stringContaining('unsuccessful event'));
    });

    it('should skip when user message is a system message', async () => {
      const { isSystemMessage } = await import('../../src/sanitizer.js');
      vi.mocked(isSystemMessage).mockReturnValueOnce(true);

      await handleAugmentation(event, ctx, config, mockLogger);
      expect(mockLogger.info).toHaveBeenCalledWith(expect.stringContaining('system message'));
    });
  });
});