import { describe, it, expect } from 'vitest';
import { isSystemMessage, cleanText } from '../src/sanitizer.js';
import type { OpenClawMessageBlock } from '../src/types.js';

describe('sanitizer', () => {
  describe('isSystemMessage', () => {
    it('should return true for empty text', () => {
      expect(isSystemMessage('')).toBe(true);
    });

    it('should return true for system startup messages', () => {
      expect(isSystemMessage('a new session was started')).toBe(true);
      expect(isSystemMessage('A NEW SESSION WAS STARTED')).toBe(true);
    });

    it('should return true for reset command messages', () => {
      expect(isSystemMessage('/new or /reset')).toBe(true);
      expect(isSystemMessage('Use /new or /reset to start fresh')).toBe(true);
    });

    it('should return true for session startup sequence', () => {
      expect(isSystemMessage('session startup sequence initiated')).toBe(true);
    });

    it('should return true for persona messages', () => {
      expect(isSystemMessage('use persona: helpful assistant')).toBe(true);
    });

    it('should return false for regular user messages', () => {
      expect(isSystemMessage('Hello, how are you?')).toBe(false);
      expect(isSystemMessage('What is the weather today?')).toBe(false);
      expect(isSystemMessage('Can you help me with this code?')).toBe(false);
    });
  });

  describe('cleanText', () => {
    describe('string content', () => {
      it('should return the string as-is when no special formatting', () => {
        const input = 'Hello, world!';
        expect(cleanText(input)).toBe('Hello, world!');
      });

      it('should extract message after last code fence', () => {
        const input = '```metadata\nsome: data\n```\nActual message here';
        expect(cleanText(input)).toBe('Actual message here');
      });

      it('should remove timestamp prefix from message', () => {
        const input = '```metadata```\n[Mon 2024-03-09 14:30 UTC] Hello there';
        expect(cleanText(input)).toBe('Hello there');
      });

      it('should handle timestamp with timezone offset', () => {
        const input = '[Tue 2024-03-10 09:15 EST] Good morning';
        expect(cleanText(input)).toBe('Good morning');
      });

      it('should remove memori_context tags', () => {
        const input = '<memori_context>Previous facts</memori_context>\nNew message';
        expect(cleanText(input)).toBe('New message');
      });

      it('should handle multiple memori_context blocks', () => {
        const input =
          '<memori_context>Block 1</memori_context>\nText\n<memori_context>Block 2</memori_context>';
        expect(cleanText(input)).toBe('Text');
      });

      it('should handle complex combination of formatting', () => {
        const input = `\`\`\`metadata
sessionId: abc-123
\`\`\`
[Wed 2024-03-11 16:45 UTC] <memori_context>Previous context</memori_context>
What is the capital of France?`;
        expect(cleanText(input)).toBe('What is the capital of France?');
      });

      it('should return empty string when only metadata present', () => {
        const input = '```metadata\ndata\n```';
        expect(cleanText(input)).toBe('');
      });

      it('should handle content without code fences', () => {
        const input = '[Thu 2024-03-12 10:00 PST] Hello';
        expect(cleanText(input)).toBe('Hello');
      });
    });

    describe('message block array content', () => {
      it('should extract text from text blocks', () => {
        const blocks: OpenClawMessageBlock[] = [
          { type: 'text', text: 'First part' },
          { type: 'text', text: 'Second part' },
        ];
        expect(cleanText(blocks)).toBe('First part\n\nSecond part');
      });

      it('should filter out non-text blocks', () => {
        const blocks: OpenClawMessageBlock[] = [
          { type: 'text', text: 'User message' },
          { type: 'thinking', thinking: 'Internal thought' },
          { type: 'tool_use', name: 'search' },
        ];
        expect(cleanText(blocks)).toBe('User message');
      });

      it('should handle blocks without explicit type but with text', () => {
        const blocks: OpenClawMessageBlock[] = [{ text: 'Message 1' }, { text: 'Message 2' }];
        expect(cleanText(blocks)).toBe('Message 1\n\nMessage 2');
      });

      it('should filter out empty text blocks', () => {
        const blocks: OpenClawMessageBlock[] = [
          { type: 'text', text: '' },
          { type: 'text', text: 'Valid message' },
          { type: 'text', text: '' },
        ];
        expect(cleanText(blocks)).toBe('Valid message');
      });

      it('should handle empty array', () => {
        expect(cleanText([])).toBe('');
      });

      it('should apply post-processing to extracted text', () => {
        const blocks: OpenClawMessageBlock[] = [
          {
            type: 'text',
            text: '```metadata```\n[Fri 2024-03-13 11:20 UTC] Clean this message',
          },
        ];
        expect(cleanText(blocks)).toBe('Clean this message');
      });
    });

    describe('edge cases', () => {
      it('should return empty string for null input', () => {
        expect(cleanText(null)).toBe('');
      });

      it('should return empty string for undefined input', () => {
        expect(cleanText(undefined)).toBe('');
      });

      it('should handle objects that are not arrays or strings', () => {
        expect(cleanText({ random: 'object' })).toBe('');
      });

      it('should handle number input', () => {
        expect(cleanText(42)).toBe('');
      });

      it('should trim whitespace from final result', () => {
        expect(cleanText('  \n  Hello  \n  ')).toBe('Hello');
      });

      it('should preserve internal whitespace', () => {
        const input = 'Hello   world\n\nNew paragraph';
        expect(cleanText(input)).toBe('Hello   world\n\nNew paragraph');
      });
    });
  });
});
