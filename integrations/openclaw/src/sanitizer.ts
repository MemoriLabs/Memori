import { OpenClawMessageBlock } from './types.js';

/**
 * Patterns for detecting OpenClaw system messages that should be ignored
 */
const SYSTEM_MESSAGE_PATTERNS = [
  'a new session was started',
  '/new or /reset',
  'session startup sequence',
  'use persona',
] as const;

/**
 * Regular expressions for cleaning text content
 */
const CLEANUP_PATTERNS = {
  /** OpenClaw's conversation metadata block */
  METADATA: /Conversation info \(untrusted metadata\):\s*```json[\s\S]*?```\s*/g,

  /** Our own memory context tags */
  MEMORI_CONTEXT: /<memori_context>[\s\S]*?<\/memori_context>\s*/g,

  /** OpenClaw's timestamp prefixes */
  TIMESTAMP:
    /^\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}| [A-Z]{3,4})?\]\s*/gm,

  /** Generic bracketed metadata */
  BRACKETS: /\[\[.*?\]\]\s*/g,
} as const;

/**
 * Type guard to check if value is an array of message blocks
 */
function isMessageBlockArray(value: unknown): value is OpenClawMessageBlock[] {
  return Array.isArray(value);
}

/**
 * Determines if a message is an OpenClaw internal system/startup prompt.
 * System messages should typically be ignored for memory operations.
 *
 * @param text - Message text to check
 * @returns true if the message is a system message
 */
export function isSystemMessage(text: string): boolean {
  if (!text) return true;

  const lowerText = text.toLowerCase();
  return SYSTEM_MESSAGE_PATTERNS.some((pattern) => lowerText.includes(pattern));
}

/**
 * Safely extracts text string from OpenClaw's multi-modal message arrays.
 * Handles string content, array content with text blocks, and ignores thinking blocks.
 *
 * @param content - Raw message content (can be string, array, or unknown)
 * @returns Extracted text string, or empty string if no text found
 */
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

/**
 * Cleans and normalizes text content from OpenClaw messages.
 *
 * @param rawContent - Raw message content to clean
 * @returns Cleaned and trimmed text string
 */
export function cleanText(rawContent: unknown): string {
  let text = extractMessageText(rawContent);

  if (!text) return '';

  text = text.replace(CLEANUP_PATTERNS.METADATA, '');
  text = text.replace(CLEANUP_PATTERNS.MEMORI_CONTEXT, '');
  text = text.replace(CLEANUP_PATTERNS.TIMESTAMP, '');
  text = text.replace(CLEANUP_PATTERNS.BRACKETS, '');

  return text.trim();
}
