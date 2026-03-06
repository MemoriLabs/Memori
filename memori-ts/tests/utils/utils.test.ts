import { describe, it, expect } from 'vitest';
import {
  stringifyContent,
  formatDate,
  extractFacts,
  extractHistory,
  extractLastUserMessage,
} from '../../src/utils/utils.js';
import { Message } from '@memorilabs/axon';

describe('Utils', () => {
  describe('formatDate', () => {
    it('should format valid ISO strings', () => {
      const input = '2023-10-25T14:30:00.000Z';
      const output = formatDate(input);
      expect(output).toBe('2023-10-25 14:30');
    });

    it('should return undefined for undefined input', () => {
      expect(formatDate(undefined)).toBeUndefined();
    });

    it('should return substring if date parsing fails but string exists', () => {
      const invalidDate = 'not-a-date-string-that-is-long';
      expect(formatDate(invalidDate)).toBe('not-a-date-strin');
    });
  });

  describe('stringifyContent', () => {
    it('should return string as is', () => {
      expect(stringifyContent('hello')).toBe('hello');
    });

    it('should handle array of strings', () => {
      expect(stringifyContent(['a', 'b'])).toBe('a\nb');
    });

    it('should handle array of objects (LLM content blocks)', () => {
      const input = [{ text: 'part1' }, { content: 'part2' }];
      expect(stringifyContent(input)).toBe('part1\npart2');
    });

    it('should handle single object', () => {
      expect(stringifyContent({ text: 'hello' })).toBe('hello');
    });

    it('should fallback to JSON stringify for unknown objects', () => {
      expect(stringifyContent({ other: 'value' })).toContain('{"other":"value"}');
    });
  });

  describe('extractFacts', () => {
    it('should extract strings directly', () => {
      const response = { facts: ['fact1', 'fact2'] };
      const result = extractFacts(response);
      expect(result).toHaveLength(2);
      expect(result[0].content).toBe('fact1');
      expect(result[0].score).toBe(1.0);
    });

    it('should extract structured objects using rank_score', () => {
      const response = {
        results: [{ content: 'fact1', rank_score: 0.8, date_created: '2023-01-01T12:00:00Z' }],
      };
      const result = extractFacts(response);
      expect(result[0].score).toBe(0.8);
      expect(result[0].dateCreated).toBeDefined();
    });

    it('should fallback to similarity if rank_score is missing', () => {
      const response = {
        results: [{ content: 'fact2', similarity: 0.65, date_created: '2023-01-01T12:00:00Z' }],
      };
      const result = extractFacts(response);
      expect(result[0].score).toBe(0.65);
    });

    it('should ignore objects without a valid content string', () => {
      const response = {
        results: [{ missing_content: 'foo' }, { content: 'valid fact', rank_score: 0.9 }],
      } as any;

      const result = extractFacts(response);
      expect(result).toHaveLength(1);
      expect(result[0].content).toBe('valid fact');
    });
  });

  describe('extractHistory', () => {
    it('should extract from messages key', () => {
      const response = { messages: ['msg1'] };
      expect(extractHistory(response)).toEqual(['msg1']);
    });

    it('should extract from conversation.messages key', () => {
      const response = { conversation: { messages: ['msg2'] } };
      expect(extractHistory(response)).toEqual(['msg2']);
    });

    it('should extract from history key', () => {
      const response = { history: ['msg3'] };
      expect(extractHistory(response)).toEqual(['msg3']);
    });

    it('should return empty array if no history found', () => {
      expect(extractHistory({})).toEqual([]);
    });
  });

  describe('extractLastUserMessage', () => {
    it('should extract the content of the last user message', () => {
      const messages: Message[] = [
        { role: 'user', content: 'first user message' },
        { role: 'assistant', content: 'assistant reply' },
        { role: 'user', content: 'second user message' },
        { role: 'system', content: 'system message' },
      ];
      expect(extractLastUserMessage(messages)).toBe('second user message');
    });

    it('should return undefined if there are no user messages', () => {
      const messages: Message[] = [
        { role: 'assistant', content: 'assistant reply' },
        { role: 'system', content: 'system message' },
      ];
      expect(extractLastUserMessage(messages)).toBeUndefined();
    });

    it('should return undefined for an empty messages array', () => {
      expect(extractLastUserMessage([])).toBeUndefined();
    });
  });
});
