import { Message } from '@memorilabs/axon';
import {
  CloudRecallResponse,
  ParsedFact,
  ParsedSummary,
  RecallItem,
  RecallSummary,
} from '../types/api.js';

/** @internal */
export function formatDate(dateStr?: string): string | undefined {
  if (!dateStr) return undefined;
  try {
    const d = new Date(dateStr);
    // If the string isn't a valid date, truncate it to 16 chars (YYYY-MM-DD HH:mm) as a best-effort
    if (isNaN(d.getTime())) return dateStr.substring(0, 16);
    return d.toISOString().replace('T', ' ').substring(0, 16);
  } catch {
    return undefined;
  }
}

/** @internal Deduplicates summaries across facts using content+date as the key. */
function collectSummariesFromFacts(facts: ParsedFact[]): ParsedSummary[] {
  const summaries: ParsedSummary[] = [];
  const seen = new Set<string>();

  for (const fact of facts) {
    if (!fact.summaries) continue;

    for (const summary of fact.summaries) {
      const key = `${summary.content}::${summary.dateCreated}`;
      if (seen.has(key)) continue;

      seen.add(key);
      summaries.push(summary);
    }
  }

  return summaries;
}

/** @internal */
export function formatSummariesFromFacts(facts: ParsedFact[]): string[] {
  return collectSummariesFromFacts(facts).map(
    (summary) => `- [${summary.dateCreated}]\n  ${summary.content}`
  );
}

/** @internal Groups cloud-returned summaries by fact ID and merges them into the fact objects. */
function attachRawSummariesToFacts(facts: RecallItem[], summaries: RecallSummary[]): RecallItem[] {
  if (summaries.length === 0) return facts;

  const summariesByFactId = new Map<number, RecallSummary[]>();

  for (const summary of summaries) {
    const summaryFactId = summary.entity_fact_id;
    const existing = summariesByFactId.get(summaryFactId) ?? [];
    existing.push(summary);
    summariesByFactId.set(summaryFactId, existing);
  }

  return facts.map((fact) => {
    if (typeof fact === 'string') return fact;

    const matchedSummaries = summariesByFactId.get(fact.id) ?? [];
    const existingSummaries = fact.summaries ?? [];

    if (existingSummaries.length === 0 && matchedSummaries.length === 0) return fact;

    return {
      ...fact,
      summaries: [...existingSummaries, ...matchedSummaries],
    };
  });
}

/** @internal Converts a raw API summary to the public `ParsedSummary` shape, filtering out entries with unparseable dates. */
function normalizeSummary(summary: RecallSummary): ParsedSummary | undefined {
  const dateCreated = formatDate(summary.date_created);
  if (!dateCreated) return undefined;

  return {
    content: summary.content,
    dateCreated,
  };
}

/**
 * Safely converts message content (string, array, or object) into a simple string.
 * Handles multi-modal arrays (e.g. OpenAI/Anthropic content blocks) by extracting text.
 * @internal
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

  if (typeof content === 'number' || typeof content === 'boolean' || typeof content === 'bigint') {
    return String(content);
  }
  return '';
}

/** @internal */
export function extractFacts(response: CloudRecallResponse): ParsedFact[] {
  const rawFacts = response.facts || response.results || response.memories || response.data || [];
  const rawSummaries = response.summaries ?? [];

  if (!Array.isArray(rawFacts)) return [];

  const factsWithSummaries = attachRawSummariesToFacts(rawFacts, rawSummaries);

  const facts: ParsedFact[] = [];

  for (const item of factsWithSummaries) {
    if (typeof item === 'string') {
      facts.push({ content: item, score: 1.0 });
    } else if (typeof item === 'object' && 'content' in item && typeof item.content === 'string') {
      let score = 0.0;
      if (typeof item.rank_score === 'number') score = item.rank_score;
      else if (typeof item.similarity === 'number') score = item.similarity;

      const summaries = item.summaries
        ?.map(normalizeSummary)
        .filter((summary): summary is ParsedSummary => summary !== undefined);

      facts.push({
        content: item.content,
        score,
        dateCreated: formatDate(item.date_created),
        summaries: summaries && summaries.length > 0 ? summaries : undefined,
      });
    }
  }
  return facts;
}

/** @internal */
export function extractHistory(response: CloudRecallResponse): unknown[] {
  const raw =
    response.messages ||
    response.conversation_messages ||
    response.history ||
    response.conversation?.messages ||
    [];

  return Array.isArray(raw) ? raw : [];
}

/** @internal */
export function extractLastUserMessage(messages: Message[]): string | undefined {
  return messages.findLast((m) => m.role === 'user')?.content;
}

/**
 * Safely converts a Node.js Buffer to a Float32Array using zero-copy memory sharing
 * when perfectly aligned, or falls back to a fast slice copy if unaligned.
 * @internal
 */
export function bufferToFloat32Array(buf: Buffer): Float32Array {
  const isAligned = buf.byteOffset % 4 === 0;
  return isAligned
    ? new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4)
    : new Float32Array(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
}
