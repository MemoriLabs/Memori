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
  id: number | string;
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
  id: number | string;
  content: string;
  rankScore?: number;
  similarity?: number;
  dateCreated?: string;
  summaries?: Array<{
    content: string;
    dateCreated: string;
    entityFactId?: number | string;
    factId?: number | string;
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
  entity_fact_id: number | string;
  fact_id: number | string;
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

/**
 * Filter parameters for the agent recall endpoint (GET /v1/agent/recall).
 * If sessionId is provided, projectId must also be provided.
 */
export interface AgentRecallParams {
  /** Filter results to memories created on or after this date/time (ISO 8601). */
  dateStart?: string;
  /** Filter results to memories created on or before this date/time (ISO 8601). */
  dateEnd?: string;
  /** Filter results to a specific project. */
  projectId?: string;
  /**
   * Filter results to a specific session.
   * Cannot be provided without projectId.
   */
  sessionId?: string;
  /** Filter results to a specific fact signal (e.g. system, user, derived). */
  signal?: string;
  /** Filter results to a specific source origin. */
  source?: string;
}

/**
 * Filter parameters for the agent recall summary endpoint (GET /v1/agent/recall/summary).
 * If sessionId is provided, projectId must also be provided.
 */
export interface AgentRecallSummaryParams {
  /** Filter summaries to memories created on or after this date/time (ISO 8601). */
  dateStart?: string;
  /** Filter summaries to memories created on or before this date/time (ISO 8601). */
  dateEnd?: string;
  /** Filter results to a specific project. */
  projectId?: string;
  /**
   * Filter results to a specific session.
   * Cannot be provided without projectId.
   */
  sessionId?: string;
}

/**
 * Raw response shape from the agent recall endpoint.
 */
export interface AgentRecallResponse {
  facts?: RecallItem[];
  results?: RecallItem[];
  memories?: RecallItem[];
  data?: RecallItem[];
}

/**
 * Raw response shape from the agent recall summary endpoint.
 */
export interface AgentRecallSummaryResponse {
  summaries?: RecallSummary[];
}

/**
 * Filter parameters for the agent compaction endpoint (GET /v1/agent/compaction).
 * If sessionId is provided, projectId must also be provided.
 */
export interface AgentCompactionParams {
  /** The external ID of the project to compact. Defaults to the current project context. */
  projectId?: string;
  /** The external ID of a specific agent session to scope the compaction to. */
  sessionId?: string;
  /** Number of recent conversation messages to include in the result. Defaults to 5. */
  numMessages?: number;
}

/** Continuation hints indicating what the agent last did and what it should do next. */
export interface CompactionContinuation {
  last_action: string;
  next_expected_action: string;
}

/** A single message from the agent's recent conversation history. */
export interface CompactionMessage {
  content: string;
  role: string;
  trace?: unknown;
  type: string;
}

/** Metadata describing when and how the compaction was generated. */
export interface CompactionMetadata {
  date: {
    execution: string;
  };
  filter: {
    project: { id: string };
    session?: { id: string };
  };
}

/** The structured state of ongoing work captured at compaction time. */
export interface CompactionState {
  active_tasks: string[];
  open_loops: string[];
  pending_results: string[];
}

/**
 * Response from the agent compaction endpoint (GET /v1/agent/compaction).
 * Contains a consolidated snapshot of the agent's long-term memory and context.
 */
export interface AgentCompactionResponse {
  continuation: CompactionContinuation;
  environment: string[];
  messages: CompactionMessage[];
  metadata: CompactionMetadata;
  standing_orders: string[];
  state: CompactionState;
  timeline?: string;
  workspace_changes: string[];
}
