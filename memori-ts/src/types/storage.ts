export interface EmbeddingRow {
  id: number | string;
  content_embedding?: Float32Array;
  content_embedding_b64?: string;
}

export interface CandidateFactRow {
  id: number | string;
  content: string;
  date_created: string;
  summaries?: Array<{ content: string; date_created: string }>;
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
        conversation_id: string;
        messages: Array<{ role: string; content: string }>;
      };
    }
  | {
      op_type: 'entity_fact.create';
      payload: {
        entity_id: string;
        facts: string[];
        conversation_id?: string | null;
        fact_embeddings?: Float32Array[];
      };
    }
  | {
      op_type: 'knowledge_graph.create';
      payload: {
        entity_id: string;
        semantic_triples: SemanticTriplePayload[];
      };
    }
  | {
      op_type: 'process_attribute.create';
      payload: {
        process_id: string;
        attributes: string[] | Record<string, string>;
      };
    }
  | {
      op_type: 'conversation.update';
      payload: {
        conversation_id: string;
        summary: string;
      };
    }
  | {
      op_type: 'upsert_fact';
      payload: {
        entity_id: string;
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
