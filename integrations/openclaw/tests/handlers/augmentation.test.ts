import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleAugmentation } from '../../src/handlers/augmentation.js';
import type {
  OpenClawEvent,
  OpenClawContext,
  MemoriPluginConfig,
  OpenClawMessage,
} from '../../src/types.js';
import type { MemoriLogger } from '../../src/utils/logger.js';

vi.mock('../../src/sanitizer.js', () => ({
  cleanText: vi.fn((content) => {
    if (typeof content === 'string') return content;
    return 'cleaned text';
  }),
  isSystemMessage: vi.fn(() => false),
}));

vi.mock('../../src/utils/index.js', () => ({
  extractContext: vi.fn(() => ({
    entityId: 'test-entity',
    sessionId: 'test-session',
    provider: 'test-provider',
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

    config = {
      apiKey: 'test-api-key',
      entityId: 'test-entity-id',
    };

    event = {
      success: true,
      messages: [
        { role: 'user', content: 'Hello, how are you?' },
        { role: 'assistant', content: "I'm doing well, thank you!" },
      ],
    };

    ctx = {
      sessionKey: 'session-123',
      messageProvider: 'test-provider',
    };
  });

  describe('successful augmentation', () => {
    it('should send user and assistant messages to memori', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          userMessage: 'Hello, how are you?',
          agentResponse: "I'm doing well, thank you!",
        })
      );
    });

    it('should include metadata in request', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            platform: 'openclaw',
            integrationSdkVersion: '0.0.1',
          }),
        })
      );
    });
  });

  describe('event validation', () => {
    it('should skip when event is unsuccessful', async () => {
      event.success = false;

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith(
        'No messages or unsuccessful event. Skipping augmentation.'
      );

      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      expect(initializeMemoriClient).not.toHaveBeenCalled();
    });

    it('should skip when messages array is empty', async () => {
      event.messages = [];

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith(
        'No messages or unsuccessful event. Skipping augmentation.'
      );
    });

    it('should skip when messages has only one message', async () => {
      event.messages = [{ role: 'user', content: 'Hello' }];

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith(
        'No messages or unsuccessful event. Skipping augmentation.'
      );
    });

    it('should skip when messages is undefined', async () => {
      event.messages = undefined;

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith(
        'No messages or unsuccessful event. Skipping augmentation.'
      );
    });
  });

  describe('message extraction', () => {
    it('should extract last user and assistant messages from recent history', async () => {
      event.messages = [
        { role: 'user', content: 'First message' },
        { role: 'assistant', content: 'First response' },
        { role: 'user', content: 'Second message' },
        { role: 'assistant', content: 'Second response' },
        { role: 'user', content: 'Third message' },
        { role: 'assistant', content: 'Third response' },
      ];

      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          userMessage: 'Third message',
          agentResponse: 'Third response',
        })
      );
    });

    it('should only consider last 5 messages', async () => {
      event.messages = [
        { role: 'user', content: 'Message 1' },
        { role: 'assistant', content: 'Response 1' },
        { role: 'user', content: 'Message 2' },
        { role: 'assistant', content: 'Response 2' },
        { role: 'user', content: 'Message 3' },
        { role: 'assistant', content: 'Response 3' },
        { role: 'user', content: 'Message 4' },
        { role: 'assistant', content: 'Response 4' },
        { role: 'user', content: 'Latest message' },
        { role: 'assistant', content: 'Latest response' },
      ];

      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          userMessage: 'Latest message',
          agentResponse: 'Latest response',
        })
      );
    });

    it('should skip system role messages', async () => {
      event.messages = [
        { role: 'system', content: 'System prompt' },
        { role: 'user', content: 'User message' },
        { role: 'assistant', content: 'Assistant response' },
      ];

      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          userMessage: 'User message',
          agentResponse: 'Assistant response',
        })
      );
    });

    it('should skip when user or assistant message is missing', async () => {
      event.messages = [
        { role: 'system', content: 'Sys' },
        { role: 'user', content: 'Only user message' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith('Missing user or assistant message. Skipping.');

      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      expect(initializeMemoriClient).not.toHaveBeenCalled();
    });

    it('should skip messages with empty cleaned content', async () => {
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockReturnValueOnce('');

      event.messages = [
        { role: 'user', content: 'Some content' },
        { role: 'assistant', content: 'Some response' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith('Missing user or assistant message. Skipping.');
    });
  });

  describe('system message filtering', () => {
    it('should skip when user message is a system message', async () => {
      const { isSystemMessage } = await import('../../src/sanitizer.js');
      vi.mocked(isSystemMessage).mockReturnValueOnce(true);

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith(
        'User message is a system message. Skipping augmentation.'
      );

      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      expect(initializeMemoriClient).not.toHaveBeenCalled();
    });
  });

  describe('synthetic responses', () => {
    it('should replace NO_REPLY with synthetic response', async () => {
      event.messages = [
        { role: 'user', content: 'Remember my name is John' },
        { role: 'assistant', content: 'NO_REPLY' },
      ];

      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          agentResponse: "Okay, I'll remember that for you.",
        })
      );

      expect(mockLogger.info).toHaveBeenCalledWith(
        'Assistant used tool-based messaging. Using synthetic response.'
      );
    });

    it('should replace SILENT_REPLY with synthetic response', async () => {
      event.messages = [
        { role: 'user', content: 'Save this for later' },
        { role: 'assistant', content: 'SILENT_REPLY' },
      ];

      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          agentResponse: "Okay, I'll remember that for you.",
        })
      );
    });
  });

  describe('thinking block removal', () => {
    it('should strip thinking blocks from assistant messages', async () => {
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementationOnce((content) => {
        if (typeof content === 'string' && content.includes('[[')) {
          return '[[This is my thought process.]]\n\nActual response here.';
        }
        return content as string;
      });

      event.messages = [
        { role: 'user', content: 'Question?' },
        {
          role: 'assistant',
          content: '[[This is my thought process.]]\n\nActual response here.',
        },
      ];

      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          agentResponse: 'Actual response here.',
        })
      );
    });
  });

  describe('metadata extraction', () => {
    it('should extract provider and model from last assistant message', async () => {
      event.messages = [
        { role: 'user', content: 'Question' },
        {
          role: 'assistant',
          content: 'Answer',
          provider: 'anthropic' as any,
          model: 'claude-3-sonnet' as any,
        },
      ];

      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            provider: 'anthropic',
            model: 'claude-3-sonnet',
          }),
        })
      );
    });

    it('should use null for missing provider and model', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          metadata: expect.objectContaining({
            provider: null,
            model: null,
          }),
        })
      );
    });
  });

  describe('error handling', () => {
    it('should handle errors gracefully', async () => {
      const { extractContext } = await import('../../src/utils/index.js');
      vi.mocked(extractContext).mockImplementationOnce(() => {
        throw new Error('Context extraction failed');
      });

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.error).toHaveBeenCalledWith(
        'Augmentation failed: Context extraction failed'
      );
    });

    it('should log non-Error objects as strings', async () => {
      const { extractContext } = await import('../../src/utils/index.js');
      vi.mocked(extractContext).mockImplementationOnce(() => {
        throw 'String error';
      });

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.error).toHaveBeenCalledWith('Augmentation failed: String error');
    });

    it('should handle API errors from memori client', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      vi.mocked(initializeMemoriClient).mockReturnValueOnce({
        augmentation: vi.fn(async () => {
          throw new Error('API connection failed');
        }),
      } as any);

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.error).toHaveBeenCalledWith('Augmentation failed: API connection failed');
    });
  });
});
