import { describe, it, expect } from 'vitest';
import {
  stringifyContent,
  formatDate,
  extractFacts,
  extractHistory,
} from '../../src/utils/utils.js';

describe('Utils', () => {
  describe('formatDate', () => {
    it('should format valid ISO strings', () => {
      const input = '2023-10-25T14:30:00.000Z';
      // Expected output depends on local time if not handled strictly,
      // but the utility replaces T with space and cuts off seconds.
      // Based on the code: d.toISOString().replace('T', ' ').substring(0, 16);
      const output = formatDate(input);
      // toISOString always returns UTC, so we expect '2023-10-25 14:30'
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

    it('should extract structured objects', () => {
      const response = {
        results: [{ content: 'fact1', rank_score: 0.8, date_created: '2023-01-01T12:00:00Z' }],
      };
      const result = extractFacts(response);
      expect(result[0].score).toBe(0.8);
      expect(result[0].dateCreated).toBeDefined();
    });
  });

  describe('extractHistory', () => {
    it('should extract from messages key', () => {
      const response = { messages: ['msg1'] };
      expect(extractHistory(response)).toEqual(['msg1']);
    });

    it('should return empty array if no history found', () => {
      expect(extractHistory({})).toEqual([]);
    });
  });
});
