import type { CandidateFactRow, EmbeddingRow, SemanticTriplePayload } from '../types/storage.js';

export type SqlBindValue =
  | string
  | number
  | boolean
  | null
  | Buffer
  | Uint8Array
  | (string | number)[];

export interface StorageAdapter {
  execute<T = Record<string, unknown>>(
    operation: string,
    binds?: SqlBindValue[]
  ): Promise<T[]> | T[];
  begin(): Promise<void> | void;
  commit(): Promise<void> | void;
  rollback(): Promise<void> | void;
  getDialect(): string;
  close(): Promise<void> | void;
}

export interface Migration {
  description: string;
  operation?: string; // single SQL statement
  operations?: string[]; // multiple statements run in sequence (e.g. CREATE TABLE + CREATE INDEX)
}

interface SchemaVersionOps {
  read(): Promise<number | null> | number | null;
  delete(): Promise<void> | void;
  create(num: number): Promise<void> | void;
}

interface SchemaOps {
  version: SchemaVersionOps;
}

interface ConversationMessageOps {
  create(
    conversationId: number | string,
    role: string,
    type: string | null,
    content: string
  ): Promise<void> | void;
}

interface ConversationMessagesOps {
  read(
    conversationId: number | string
  ): Promise<Array<{ role: string; content: string }>> | Array<{ role: string; content: string }>;
}

interface ConversationOps {
  create(
    sessionId: number | string | null,
    timeoutMinutes: number
  ): Promise<number | string | null> | number | string | null;
  update(id: number | string | null, summary: string): unknown;
}

interface EntityOps {
  create(externalId: string | number): Promise<number | string | null> | number | string | null;
}

interface EntityFactOps {
  create(
    entityId: number | string,
    facts: string[],
    factEmbeddings?: Float32Array[],
    conversationId?: number | string | null
  ): unknown;
  createWithoutEmbedding(entityId: number | string, content: string): Promise<void> | void;
  getEmbeddings(
    entityId: string | number,
    limit?: number
  ): Promise<EmbeddingRow[]> | EmbeddingRow[];
  getFactsByIds(factIds: (string | number)[]): Promise<CandidateFactRow[]> | CandidateFactRow[];
}

interface KnowledgeGraphOps {
  create(entityId: number | string, semanticTriples: SemanticTriplePayload[]): unknown;
}

interface ProcessOps {
  create(externalId: string | number): Promise<number | string | null> | number | string | null;
}

interface ProcessAttributeOps {
  create(processId: number | string, attributes: string[]): unknown;
}

interface SessionOps {
  create(
    uuid: string | number | null,
    entityId: number | string | null,
    processId: number | string | null
  ): Promise<number | string | null> | number | string | null;
}

export abstract class BaseDriver {
  /** When true, the Builder will issue a ROLLBACK if reading the schema version fails. Needed for PostgreSQL/MySQL but not SQLite. */
  public abstract readonly requiresRollbackOnError: boolean;
  public abstract readonly migrations: Partial<Record<number, Migration[]>>;

  constructor(protected readonly conn: StorageAdapter) {}

  // The `!` (definite assignment) on each property is safe because all concrete
  // subclasses initialize them in their constructor before any method is called.
  public conversation!: ConversationOps;
  public conversationMessage!: ConversationMessageOps;
  public conversationMessages!: ConversationMessagesOps;
  public entity!: EntityOps;
  public entityFact!: EntityFactOps;
  public knowledgeGraph!: KnowledgeGraphOps;
  public process!: ProcessOps;
  public processAttribute!: ProcessAttributeOps;
  public schema!: SchemaOps;
  public session!: SessionOps;
}
