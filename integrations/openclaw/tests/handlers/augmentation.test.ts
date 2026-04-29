import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleAugmentation } from '../../src/handlers/augmentation.js';
import type { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from '../../src/types.js';
import type { MemoriLogger } from '../../src/utils/logger.js';
import { SDK_VERSION } from '../../src/version.js';
import { MESSAGE_CONSTANTS } from '../../src/constants.js';

vi.mock('../../src/sanitizer.js', () => ({
  cleanText: vi.fn((content) => {
    if (typeof content === 'string') return content;
    return '';
  }),
  isSystemMessage: vi.fn(() => false),
  extractContentType: vi.fn(() => 'text'),
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
        { role: 'assistant', content: 'Hi' },
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
            { type: 'toolCall', id: '2', name: 'null_args', arguments: null },
          ],
        },
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
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : 'response text'));

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          userMessage: expect.objectContaining({ content: 'Hello' }),
          agentResponse: expect.objectContaining({ content: 'Hi' }),
        })
      );

      const payload = vi.mocked(client.augmentation).mock.calls[0][0];
      expect(payload).not.toHaveProperty('trace');
    });

    it('should include LLM metadata in request', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : 'text'));

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
            tools: [
              { name: 'get_time', args: { timezone: 'UTC' }, result: '2024-03-21T10:00:00Z' },
            ],
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
          agentResponse: expect.objectContaining({ content: MESSAGE_CONSTANTS.SYNTHETIC_RESPONSE }),
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

    it('should skip when messages array is undefined', async () => {
      event.messages = undefined;
      await handleAugmentation(event, ctx, config, mockLogger);
      expect(mockLogger.info).toHaveBeenCalledWith(expect.stringContaining('No messages'));
    });

    it('should skip when messages has fewer than 2 entries', async () => {
      event.messages = [{ role: 'user', content: 'hello' }];
      await handleAugmentation(event, ctx, config, mockLogger);
      expect(mockLogger.info).toHaveBeenCalledWith(expect.stringContaining('No messages'));
    });

    it('should skip when no user message is found in the turn', async () => {
      event.messages = [
        { role: 'assistant', content: 'first response' },
        { role: 'assistant', content: 'second response' },
      ];
      await handleAugmentation(event, ctx, config, mockLogger);
      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Missing user or assistant')
      );
    });

    it('should skip when no assistant message is found in the turn', async () => {
      event.messages = [
        { role: 'user', content: 'question one' },
        { role: 'user', content: 'question two' },
      ];
      await handleAugmentation(event, ctx, config, mockLogger);
      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Missing user or assistant')
      );
    });

    it('should skip when user message is a system message', async () => {
      const { isSystemMessage } = await import('../../src/sanitizer.js');
      vi.mocked(isSystemMessage).mockReturnValueOnce(true);

      await handleAugmentation(event, ctx, config, mockLogger);
      expect(mockLogger.info).toHaveBeenCalledWith(expect.stringContaining('system message'));
    });
  });

  describe('message content edge cases', () => {
    it('should strip [[...]] prefix from assistant response', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      event.messages = [
        { role: 'user', content: 'What is 2+2?' },
        { role: 'assistant', content: '[[INTERNAL_FLAG]] Four' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({ agentResponse: expect.objectContaining({ content: 'Four' }) })
      );
    });

    it('should use synthetic response for NO_REPLY assistant content', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : ''));

      event.messages = [
        { role: 'user', content: 'Update the state' },
        { role: 'assistant', content: MESSAGE_CONSTANTS.NO_REPLY },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          agentResponse: expect.objectContaining({ content: MESSAGE_CONSTANTS.SYNTHETIC_RESPONSE }),
        })
      );
    });
  });

  describe('Anthropic tool_use format', () => {
    it('should extract tool calls in tool_use format (block.input)', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : ''));

      event.messages = [
        { role: 'user', content: 'Search for something' },
        {
          role: 'assistant',
          content: [
            {
              type: 'tool_use',
              id: 'toolu_1',
              name: 'web_search',
              input: { query: 'hello world' },
            },
          ],
        },
        { role: 'toolResult', toolCallId: 'toolu_1', content: 'search result' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          trace: {
            tools: [
              { name: 'web_search', args: { query: 'hello world' }, result: 'search result' },
            ],
          },
        })
      );
    });
  });

  describe('multi-step agent turns', () => {
    it('should collect tool calls across multiple assistant messages in one turn', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : ''));

      event.messages = [
        { role: 'user', content: 'Do multiple things' },
        {
          role: 'assistant',
          content: [{ type: 'toolCall', id: 'step1', name: 'action_one', arguments: { x: 1 } }],
        },
        { role: 'toolResult', toolCallId: 'step1', content: 'result one' },
        {
          role: 'assistant',
          content: [{ type: 'toolCall', id: 'step2', name: 'action_two', arguments: { y: 2 } }],
        },
        { role: 'toolResult', toolCallId: 'step2', content: 'result two' },
        { role: 'assistant', content: 'All done!' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      const tools = vi.mocked(client.augmentation).mock.calls[0][0].trace?.tools;

      // Both tool calls should be captured in chronological order
      expect(tools).toHaveLength(2);
      expect(tools[0]).toEqual({ name: 'action_one', args: { x: 1 }, result: 'result one' });
      expect(tools[1]).toEqual({ name: 'action_two', args: { y: 2 }, result: 'result two' });
    });
  });

  describe('branch coverage', () => {
    it('should skip non-tool blocks when iterating mixed assistant content', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockImplementation((c) => (typeof c === 'string' ? c : ''));

      event.messages = [
        { role: 'user', content: 'question' },
        {
          role: 'assistant',
          content: [
            { type: 'text', text: 'thinking...' }, // non-tool block — covers line 82 false branch
            { type: 'toolCall', id: 'c1', name: 'search', arguments: { q: 'test' } },
          ],
        },
        { role: 'toolResult', toolCallId: 'c1', content: 'result' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      const tools = vi.mocked(client.augmentation).mock.calls[0][0].trace?.tools;
      expect(tools).toHaveLength(1);
      expect(tools[0].name).toBe('search');
    });

    it('should skip when user message content cleans to empty', async () => {
      // Covers line 102 false branch: parseUserMessage returns null.
      // The default mock returns strings as-is, so '' content → cleanText('') = '' → null.
      event.messages = [
        { role: 'user', content: '' },
        { role: 'assistant', content: 'answer' },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Missing user or assistant')
      );
    });

    it('should skip when assistant has empty content and no tool calls', async () => {
      // Covers line 137 else-if false branch: cleanedContent empty AND no tools
      event.messages = [
        { role: 'user', content: 'question' },
        { role: 'assistant', content: [] },
      ];

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.info).toHaveBeenCalledWith(
        expect.stringContaining('Missing user or assistant')
      );
    });

    it('should log stringified non-Error values thrown in catch block', async () => {
      // Covers line 234 String(err) branch
      const { extractContext } = await import('../../src/utils/index.js');
      vi.mocked(extractContext).mockImplementationOnce(() => {
        throw 'plain string error';
      });

      await handleAugmentation(event, ctx, config, mockLogger);

      expect(mockLogger.error).toHaveBeenCalledWith(
        expect.stringContaining('Augmentation failed: plain string error')
      );
    });
  });

  describe('max context messages', () => {
    it('should only consider the last 20 messages when parsing the turn', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');

      // 22 messages total: the first pair (indices 0-1) falls outside the 20-message window
      const messages = [
        { role: 'user', content: 'excluded user message' },
        { role: 'assistant', content: 'excluded assistant response' },
        // 18 padding assistant messages to push the first pair out of the window
        ...Array.from({ length: 18 }, (_, i) => ({ role: 'assistant', content: `pad ${i}` })),
        { role: 'user', content: 'included user message' },
        { role: 'assistant', content: 'included assistant response' },
      ];

      event.messages = messages;

      await handleAugmentation(event, ctx, config, mockLogger);

      const client = vi.mocked(initializeMemoriClient).mock.results[0].value as any;
      expect(client.augmentation).toHaveBeenCalledWith(
        expect.objectContaining({
          userMessage: expect.objectContaining({ content: 'included user message' }),
          agentResponse: expect.objectContaining({ content: 'included assistant response' }),
        })
      );
    });
  });
});
