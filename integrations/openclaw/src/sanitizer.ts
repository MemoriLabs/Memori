import { OpenClawMessageBlock } from './types.js';

const SYSTEM_MESSAGE_PATTERNS = [
  'a new session was started',
  '/new or /reset',
  'session startup sequence',
  'use persona',
] as const;

const TIMESTAMP_PREFIX_REGEX =
  /^\[[A-Z][a-z]{2}\s+\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s+[A-Z]{2,4}\]\s*/;

function isMessageBlockArray(value: unknown): value is OpenClawMessageBlock[] {
  return Array.isArray(value) && value.every((item) => item !== null && typeof item === 'object');
}

export function isSystemMessage(text: string): boolean {
  if (!text) return true;
  const lowerText = text.toLowerCase();
  return SYSTEM_MESSAGE_PATTERNS.some((pattern) => lowerText.includes(pattern));
}

/**
 * Extracts the actual user message from OpenClaw's formatted content.
 *
 * OpenClaw wraps metadata in markdown code fences (triple tick).
 * The actual message is always after the LAST closing fence.
 * The message might aslo contain a timestamp prefix like: [Day YYYY-MM-DD HH:MM TZ], that will need to be removed
 */
function extractRawUserMessage(content: string): string {
  let message = content;

  if (message.includes('```')) {
    const lastFenceIndex = message.lastIndexOf('```');
    if (lastFenceIndex !== -1) {
      message = message.substring(lastFenceIndex + 3).trim();
    }
  }

  const timestampMatch = message.match(TIMESTAMP_PREFIX_REGEX);
  if (timestampMatch) {
    message = message.substring(timestampMatch[0].length).trim();
  }

  return message;
}

function extractMessageText(content: unknown): string {
  if (!content) return '';

  if (typeof content === 'string') {
    return content;
  }

  if (isMessageBlockArray(content)) {
    return content
      .filter((block) => (block.type === 'text' || typeof block.text === 'string') && block.text)
      .map((block) => block.text)
      .join('\n\n');
  }

  return '';
}

export function cleanText(rawContent: unknown): string {
  let text = extractMessageText(rawContent);

  if (!text) return '';

  text = extractRawUserMessage(text);

  text = text.replace(/<memori_context>[\s\S]*?<\/memori_context>\s*/g, '');

  return text.trim();
}
