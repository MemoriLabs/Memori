import { describe, it, expect, vi, beforeEach } from 'vitest';
import { handleRecall } from '../../src/handlers/recall.js';
import type { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from '../../src/types.js';
import type { MemoriLogger } from '../../src/utils/logger.js';

vi.mock('../../src/sanitizer.js', () => ({
  cleanText: vi.fn((text) => text),
  isSystemMessage: vi.fn(() => false),
}));

vi.mock('../../src/utils/index.js', () => ({
  extractContext: vi.fn(() => ({
    entityId: 'test-entity',
    sessionId: 'test-session',
    provider: 'test-provider',
  })),
  initializeMemoriClient: vi.fn(() => ({
    recall: vi.fn(async () => '<memori_context>Relevant memories</memori_context>'),
  })),
}));

describe('handlers/recall', () => {
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
      prompt: 'What is the weather today?',
    };

    ctx = {
      sessionKey: 'session-123',
      messageProvider: 'test-provider',
    };
  });

  describe('successful recall', () => {
    it('should return prependContext when memories are found', async () => {
      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toEqual({
        prependContext: '<memori_context>Relevant memories</memori_context>',
      });
    });

    it('should call extractContext with correct parameters', async () => {
      const { extractContext } = await import('../../src/utils/index.js');
      await handleRecall(event, ctx, config, mockLogger);

      expect(extractContext).toHaveBeenCalledWith(event, ctx, 'test-entity-id');
    });

    it('should call initializeMemoriClient with correct parameters', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      await handleRecall(event, ctx, config, mockLogger);

      expect(initializeMemoriClient).toHaveBeenCalledWith('test-api-key', {
        entityId: 'test-entity',
        sessionId: 'test-session',
        provider: 'test-provider',
      });
    });
  });

  describe('no memories found', () => {
    it('should return undefined when memori returns empty string', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      vi.mocked(initializeMemoriClient).mockReturnValueOnce({
        recall: vi.fn(async () => ''),
      } as any);

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeUndefined();
      expect(mockLogger.info).toHaveBeenCalledWith('No relevant memories found.');
    });

    it('should return undefined when memori returns null', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      vi.mocked(initializeMemoriClient).mockReturnValueOnce({
        recall: vi.fn(async () => null),
      } as any);

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeUndefined();
    });
  });

  describe('prompt validation', () => {
    it('should abort when prompt is empty after cleaning', async () => {
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockReturnValueOnce('');

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeUndefined();
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Prompt too short or is a system message. Aborting recall.'
      );
    });

    it('should abort when prompt is too short', async () => {
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockReturnValueOnce('a');

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeUndefined();
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Prompt too short or is a system message. Aborting recall.'
      );
    });

    it('should abort when prompt is a system message', async () => {
      const { isSystemMessage } = await import('../../src/sanitizer.js');
      vi.mocked(isSystemMessage).mockReturnValueOnce(true);

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeUndefined();
      expect(mockLogger.info).toHaveBeenCalledWith(
        'Prompt too short or is a system message. Aborting recall.'
      );
    });

    it('should proceed when prompt meets minimum length', async () => {
      const { cleanText } = await import('../../src/sanitizer.js');
      vi.mocked(cleanText).mockReturnValueOnce('Hello');

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeDefined();
    });
  });

  describe('error handling', () => {
    it('should handle errors gracefully and return undefined', async () => {
      const { extractContext } = await import('../../src/utils/index.js');
      vi.mocked(extractContext).mockImplementationOnce(() => {
        throw new Error('Context extraction failed');
      });

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeUndefined();
      expect(mockLogger.error).toHaveBeenCalledWith('Recall failed: Context extraction failed');
    });

    it('should log non-Error objects as strings', async () => {
      const { extractContext } = await import('../../src/utils/index.js');
      vi.mocked(extractContext).mockImplementationOnce(() => {
        throw 'String error';
      });

      await handleRecall(event, ctx, config, mockLogger);

      expect(mockLogger.error).toHaveBeenCalledWith('Recall failed: String error');
    });

    it('should handle API errors from memori client', async () => {
      const { initializeMemoriClient } = await import('../../src/utils/index.js');
      vi.mocked(initializeMemoriClient).mockReturnValueOnce({
        recall: vi.fn(async () => {
          throw new Error('API connection failed');
        }),
      } as any);

      const result = await handleRecall(event, ctx, config, mockLogger);

      expect(result).toBeUndefined();
      expect(mockLogger.error).toHaveBeenCalledWith('Recall failed: API connection failed');
    });
  });

  describe('edge cases', () => {
    it('should handle missing prompt gracefully', async () => {
      const eventWithoutPrompt: OpenClawEvent = {};

      const result = await handleRecall(eventWithoutPrompt, ctx, config, mockLogger);

      expect(result).toBeUndefined();
    });

    it('should handle null prompt', async () => {
      const eventWithNullPrompt: OpenClawEvent = { prompt: null as any };

      const result = await handleRecall(eventWithNullPrompt, ctx, config, mockLogger);

      expect(result).toBeUndefined();
    });

    it('should call cleanText with the prompt', async () => {
      const { cleanText } = await import('../../src/sanitizer.js');

      await handleRecall(event, ctx, config, mockLogger);

      expect(cleanText).toHaveBeenCalledWith('What is the weather today?');
    });
  });
});
