/**
 * Represents a single recalled item from the backend.
 * Can be a simple string or a structured object with scoring metadata.
 * @internal
 */
export interface RecallObject {
  content: string;
  rank_score?: number;
  similarity?: number;
  date_created?: string;
}

/**
 * @internal
 */
export type RecallItem = string | RecallObject;

/**
 * Raw response shape from the Memori Cloud API.
 * @internal
 */
export interface CloudRecallResponse {
  // The API might return the list of facts under any of these keys
  facts?: RecallItem[];
  results?: RecallItem[];
  memories?: RecallItem[];
  data?: RecallItem[];

  // History fields
  messages?: unknown[];
  conversation_messages?: unknown[];
  history?: unknown[];
  conversation?: { messages?: unknown[] };
}

/**
 * A normalized memory fact returned to the user.
 */
export interface ParsedFact {
  /**
   * The actual text content of the memory or fact.
   */
  content: string;

  /**
   * The relevance score of this fact to the query (0.0 to 1.0).
   * Higher is more relevant.
   */
  score: number;

  /**
   * The ISO timestamp (YYYY-MM-DD HH:mm) when this memory was originally created.
   * Undefined if the backend did not return temporal data.
   */
  dateCreated?: string;
}
