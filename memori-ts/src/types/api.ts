/**
 * Request payload for retrieving facts from the Rust core.
 */
export interface RetrievalRequest {
  entity_id: string;
  query_text: string;
  dense_limit: number;
  limit: number;
}

/**
 * Represents a single recalled item from the backend.
 * Can be a simple string or a structured object with scoring metadata.
 * @internal
 */
export interface RecallObject {
  id: number;
  content: string;
  rank_score?: number;
  similarity?: number;
  date_created?: string;
  summaries?: RecallSummary[];
}

/**
 * Single row from the native N-API `retrieve` response (camelCase before mapping to `RecallObject`).
 */
export interface NapiRecallRow {
  id: number;
  content: string;
  rankScore?: number;
  similarity?: number;
  dateCreated?: string;
  summaries?: Array<{
    content: string;
    dateCreated: string;
    entityFactId?: number;
    factId?: number;
  }>;
}

/**
 * @internal
 */
export type RecallItem = string | RecallObject;

/**
 * Represents a summary associated with a recalled fact.
 * @internal
 */
export interface RecallSummary {
  content: string;
  date_created: string;
  entity_fact_id: number;
  fact_id: number;
}

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
  summaries?: RecallSummary[];

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

  /**
   * Summaries associated with this fact, if provided by the backend.
   */
  summaries?: ParsedSummary[];
}

/**
 * A normalized summary returned alongside a fact.
 */
export interface ParsedSummary {
  /**
   * The actual summary text.
   */
  content: string;

  /**
   * The ISO timestamp (YYYY-MM-DD HH:mm) when this summary was created.
   * Undefined if the backend did not return temporal data.
   */
  dateCreated: string;
}
