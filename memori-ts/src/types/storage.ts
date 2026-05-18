export interface EmbeddingRow {
  id: number | string;
  content_embedding?: Float32Array;
  content_embedding_b64?: string;
}

export interface CandidateSummaryRow {
  content: string;
  date_created: string;
  project_id?: string | null;
  session_id?: string | null;
  conversation_id?: number | string | null;
  source?: string | null;
  signal?: string | null;
}

export interface CandidateFactRow {
  id: number | string;
  content: string;
  date_created: string;
  summaries?: CandidateSummaryRow[];
}

export interface DetailedConversationMessageRow {
  role: string;
  content: string;
  type?: string | null;
  trace?: string | null;
  source?: string | null;
  signal?: string | null;
  date_created?: string;
}

export interface ConversationSummaryRow {
  conversation_id: number | string;
  project_id?: string | null;
  session_id?: string | null;
  content: string;
  date_created: string;
}

export interface SemanticTriplePayload {
  subject: string | { name: string; type: string };
  predicate: string;
  object: string | { name: string; type: string };
}

/**
 * Strict Discriminated Union representing all possible write operations
 * emitted by the Rust core's augmentation pipeline.
 */
export type WriteOp =
  | {
      op_type: 'conversation_message.create';
      payload: {
        conversation_id: string | number;
        project_id?: string | null;
        entity_id?: string | number | null;
        process_id?: string | number | null;
        messages: Array<{
          role: string;
          content: string;
          type?: string | null;
          trace?: unknown;
          source?: string | null;
          signal?: string | null;
        }>;
      };
    }
  | {
      op_type: 'entity_fact.create';
      payload: {
        entity_id: string | number;
        facts: string[];
        conversation_id?: string | number | null;
        fact_embeddings?: Float32Array[];
      };
    }
  | {
      op_type: 'knowledge_graph.create';
      payload: {
        entity_id: string | number;
        semantic_triples: SemanticTriplePayload[];
      };
    }
  | {
      op_type: 'process_attribute.create';
      payload: {
        process_id: string | number;
        attributes: string[] | Record<string, string>;
      };
    }
  | {
      op_type: 'conversation.update';
      payload: {
        conversation_id: string | number;
        summary: string;
      };
    }
  | {
      op_type: 'upsert_fact';
      payload: {
        entity_id: string | number;
        content: string;
        metadata?: unknown;
      };
    };

export interface WriteBatch {
  ops: WriteOp[];
}

export interface WriteAck {
  written_ops: number;
}

/**
 * The core contract that all Database Adapters must fulfill to power Memori locally.
 */
export interface StorageBridge {
  fetchEmbeddings(entityId: string, limit: number): Promise<EmbeddingRow[]> | EmbeddingRow[];
  fetchFactsByIds(ids: (number | string)[]): Promise<CandidateFactRow[]> | CandidateFactRow[];
  writeBatch(batch: WriteBatch): Promise<WriteAck> | WriteAck;
  getConversationHistory(
    sessionId: string
  ): Promise<Array<{ role: string; content: string }>> | Array<{ role: string; content: string }>;
}
