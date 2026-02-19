import { HostedRecallResponse, ParsedFact } from '../types/api.js';

export function formatDate(dateStr?: string): string | undefined {
  if (!dateStr) return undefined;
  try {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr.substring(0, 16);
    return d.toISOString().replace('T', ' ').substring(0, 16);
  } catch {
    return undefined;
  }
}

/**
 * Safely converts message content (string, array, or object) into a simple string.
 * Handles multi-modal arrays (e.g. OpenAI/Anthropic content blocks) by extracting text.
 */
export function stringifyContent(content: unknown): string {
  if (!content) return '';
  if (typeof content === 'string') return content;

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === 'string') return part;
        if (part && typeof part === 'object') {
          const obj = part as Record<string, unknown>;
          const text = obj.text ?? obj.content;
          return typeof text === 'string' ? text : '';
        }
        return '';
      })
      .join('\n');
  }

  if (typeof content === 'object') {
    const obj = content as Record<string, unknown>;
    const text = obj.text ?? obj.content;
    return typeof text === 'string' ? text : JSON.stringify(content);
  }

  return String(content as string | number | boolean);
}

export function extractFacts(response: HostedRecallResponse): ParsedFact[] {
  const raw = response.facts || response.results || response.memories || response.data || [];

  if (!Array.isArray(raw)) return [];

  const facts: ParsedFact[] = [];

  for (const item of raw) {
    if (typeof item === 'string') {
      facts.push({ content: item, score: 1.0 });
    } else if (typeof item === 'object' && 'content' in item && typeof item.content === 'string') {
      let score = 0.0;
      if (typeof item.rank_score === 'number') score = item.rank_score;
      else if (typeof item.similarity === 'number') score = item.similarity;

      facts.push({
        content: item.content,
        score,
        dateCreated: formatDate(item.date_created),
      });
    }
  }
  return facts;
}

export function extractHistory(response: HostedRecallResponse): unknown[] {
  const raw =
    response.messages ||
    response.conversation_messages ||
    response.history ||
    response.conversation?.messages ||
    [];

  return Array.isArray(raw) ? raw : [];
}
