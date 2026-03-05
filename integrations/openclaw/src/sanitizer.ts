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

  /** OpenClaw's timestamp prefixes - matches formats like:
   * [Tue 2026-03-03 18:32 UTC]
   * [Sun 2026-02-22 16:59:12+00:00]
   * [Mon 2025-12-31 23:59 EST]
   */
  TIMESTAMP:
    /^\[[A-Z][a-z]{2} \d{4}-\d{2}-\d{2} \d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}| [A-Z]{3,4})?\]\s*/gm,

  /** Generic bracketed metadata */
  BRACKETS: /\[\[.*?\]\]\s*/g,
} as const;

/**
 * Type guard to check if value is a string
 */
function isString(value: unknown): value is string {
  return typeof value === 'string';
}

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
 *
 * @example
 * ```typescript
 * isSystemMessage("A new session was started") // true
 * isSystemMessage("Hello, how are you?") // false
 * ```
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
 *
 * @example
 * ```typescript
 * extractMessageText("Hello") // "Hello"
 * extractMessageText([{ type: "text", text: "Hello" }]) // "Hello"
 * extractMessageText([{ type: "thinking", thinking: "..." }]) // ""
 * ```
 */
function extractMessageText(content: unknown): string {
  if (!content) return '';

  if (isString(content)) {
    return content;
  }

  if (isMessageBlockArray(content)) {
    // Find the text block (ignoring thinking/metadata blocks)
    const textBlock = content.find((block) => block.type === 'text' || block.text);
    return textBlock && isString(textBlock.text) ? textBlock.text : '';
  }

  return '';
}

/**
 * Cleans and normalizes text content from OpenClaw messages.
 *
 * This function performs the following operations:
 * 1. Extracts text from multi-modal content
 * 2. Removes OpenClaw metadata blocks
 * 3. Removes injected memory context tags (prevents loops)
 * 4. Removes timestamp prefixes
 * 5. Removes bracketed metadata
 *
 * @param rawContent - Raw message content to clean
 * @returns Cleaned and trimmed text string
 *
 * @example
 * ```typescript
 * const raw = "[Sun 2026-03-03 16:00 UTC] <memori_context>...</memori_context>Hello";
 * cleanText(raw) // "Hello"
 * ```
 */
export function cleanText(rawContent: unknown): string {
  // 1. Extract text from multi-modal content
  let text = extractMessageText(rawContent);

  if (!text) return '';

  // 2. Strip OpenClaw's hidden metadata JSON block
  text = text.replace(CLEANUP_PATTERNS.METADATA, '');

  // 3. Strip our own injected memory context tags (prevents feedback loops)
  text = text.replace(CLEANUP_PATTERNS.MEMORI_CONTEXT, '');

  // 4. Strip OpenClaw's timestamp prefixes
  text = text.replace(CLEANUP_PATTERNS.TIMESTAMP, '');

  // 5. Strip generic bracketed metadata
  text = text.replace(CLEANUP_PATTERNS.BRACKETS, '');

  return text.trim();
}
