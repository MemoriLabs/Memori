import { randomUUID, createHash } from 'node:crypto';
import { StorageAdapter, BaseDriver, SqlBindValue } from '../base.js';
import { postgresqlMigrations } from '../migrations/postgresql.js';
import { Registry } from '../registry.js';
import { CandidateFactRow, SemanticTriplePayload } from '../../types/storage.js';
import { bufferToFloat32Array } from '../../utils/utils.js';

function generateUniq(inputs: string[]): string {
  const hash = createHash('sha256');
  for (const input of inputs) {
    hash.update(input);
  }
  return hash.digest('hex');
}

function formatEmbeddingForDb(embedding: Float32Array): Buffer {
  return Buffer.from(embedding.buffer, embedding.byteOffset, embedding.byteLength);
}

class ConversationMessage {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(
    conversationId: number | string,
    role: string,
    type: string | null,
    content: string
  ): Promise<void> {
    await this.conn.execute(
      `INSERT INTO memori_conversation_message(uuid, conversation_id, role, type, content) VALUES ($1, $2, $3, $4, $5)`,
      [randomUUID(), conversationId, role, type, content]
    );
  }
}

class ConversationMessages {
  constructor(private readonly conn: StorageAdapter) {}
  public async read(
    conversationId: number | string
  ): Promise<Array<{ role: string; content: string }>> {
    const results = await this.conn.execute<{ role: string; content: string }>(
      `SELECT role, content FROM memori_conversation_message WHERE conversation_id = $1`,
      [conversationId]
    );
    return results.map((row) => ({ content: row.content, role: row.role }));
  }
}

class Conversation {
  constructor(
    private readonly conn: StorageAdapter,
    public readonly message: ConversationMessage,
    public readonly messages: ConversationMessages
  ) {}
  public async create(
    sessionId: number | string,
    timeoutMinutes: number
  ): Promise<number | string | null> {
    const existing = await this.conn.execute<{ id: number | string; last_activity: string }>(
      `SELECT c.id, COALESCE(MAX(m.date_created), c.date_created) as last_activity FROM memori_conversation c LEFT JOIN memori_conversation_message m ON m.conversation_id = c.id WHERE c.session_id = $1 GROUP BY c.id, c.date_created`,
      [sessionId]
    );
    if (existing.length > 0) {
      const result = await this.conn.execute<{ minutes_since_activity: number }>(
        `SELECT EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - $1::timestamp)) / 60 as minutes_since_activity`,
        [existing[0].last_activity]
      );
      if (result.length > 0 && result[0].minutes_since_activity <= timeoutMinutes)
        return existing[0].id;
    }
    await this.conn.execute(
      `INSERT INTO memori_conversation(uuid, session_id) VALUES ($1, $2) ON CONFLICT DO NOTHING`,
      [randomUUID(), sessionId]
    );
    const newConv = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_conversation WHERE session_id = $1`,
      [sessionId]
    );
    return newConv.length > 0 ? newConv[0].id : null;
  }
  public async update(id: number | string, summary: string): Promise<this> {
    if (!summary) return this;
    await this.conn.execute(`UPDATE memori_conversation SET summary = $1 WHERE id = $2`, [
      summary,
      id,
    ]);
    return this;
  }
}

class Entity {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(externalId: string | number): Promise<number | string | null> {
    await this.conn.execute(
      `INSERT INTO memori_entity(uuid, external_id) VALUES ($1, $2) ON CONFLICT DO NOTHING`,
      [randomUUID(), externalId]
    );
    const res = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_entity WHERE external_id = $1`,
      [externalId]
    );
    return res.length > 0 ? res[0].id : null;
  }
}

class EntityFact {
  constructor(private readonly conn: StorageAdapter) {}

  public async create(
    entityId: number | string,
    facts: string[],
    factEmbeddings?: Float32Array[],
    conversationId?: number | string | null
  ): Promise<this> {
    if (facts.length === 0) return this;

    const CHUNK_SIZE = 2000; // Safe threshold for PostgreSQL parameter limits

    // 1. Build and Execute the Bulk Insert Arrays in chunks
    for (let i = 0; i < facts.length; i += CHUNK_SIZE) {
      const chunkFacts = facts.slice(i, i + CHUNK_SIZE);
      const values: SqlBindValue[] = [];
      const placeholders: string[] = [];
      let paramIdx = 1;

      for (let j = 0; j < chunkFacts.length; j++) {
        const globalIdx = i + j;
        const fact = chunkFacts[j];
        const embedding =
          factEmbeddings && globalIdx < factEmbeddings.length
            ? factEmbeddings[globalIdx]
            : new Float32Array(0);

        if (embedding.length === 0) continue;

        const embeddingFormatted = formatEmbeddingForDb(embedding);
        const uniq = generateUniq([fact]); // Still using sync node:crypto

        // ($1, $2, $3, $4, 1, CURRENT_TIMESTAMP, $5)
        placeholders.push(
          `($${paramIdx++}, $${paramIdx++}, $${paramIdx++}, $${paramIdx++}, 1, CURRENT_TIMESTAMP, $${paramIdx++})`
        );
        values.push(randomUUID(), entityId, fact, embeddingFormatted, uniq);
      }

      if (values.length > 0) {
        // Execute chunk Bulk Insert
        await this.conn.execute(
          `INSERT INTO memori_entity_fact(uuid, entity_id, content, content_embedding, num_times, date_last_time, uniq) 
           VALUES ${placeholders.join(', ')} 
           ON CONFLICT (entity_id, uniq) DO UPDATE 
           SET num_times = memori_entity_fact.num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
          values
        );
      }
    }

    // 2. Handle Conversation Linking (if applicable)
    if (conversationId) {
      // Fetch the IDs we just inserted/updated
      const insertedFacts = await this.conn.execute<{ id: number | string }>(
        `SELECT id FROM memori_entity_fact WHERE entity_id = $1 ORDER BY date_last_time DESC LIMIT $2`,
        [entityId, facts.length]
      );

      if (insertedFacts.length > 0) {
        const linkValues: SqlBindValue[] = [];
        const linkPlaceholders: string[] = [];
        let linkIdx = 1;

        for (const row of insertedFacts) {
          linkPlaceholders.push(`($${linkIdx++}, $${linkIdx++}, $${linkIdx++}, $${linkIdx++})`);
          linkValues.push(randomUUID(), entityId, row.id, conversationId);
        }

        await this.conn.execute(
          `INSERT INTO memori_entity_fact_mention(uuid, entity_id, fact_id, conversation_id) 
           VALUES ${linkPlaceholders.join(', ')} 
           ON CONFLICT (entity_id, fact_id, conversation_id) DO NOTHING`,
          linkValues
        );
      }
    }
    return this;
  }

  public async createWithoutEmbedding(entityId: number | string, content: string): Promise<void> {
    const uniq = generateUniq([content]);
    await this.conn.execute(
      `INSERT INTO memori_entity_fact(uuid, entity_id, content, content_embedding, num_times, date_last_time, uniq) VALUES ($1, $2, $3, $4, 1, CURRENT_TIMESTAMP, $5) ON CONFLICT (entity_id, uniq) DO UPDATE SET num_times = memori_entity_fact.num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
      [randomUUID(), entityId, content, Buffer.alloc(0), uniq]
    );
  }

  public async getEmbeddings(entityId: string | number, limit: number = 1000) {
    const results = await this.conn.execute<{
      id: number | string;
      content_embedding: Buffer | null;
    }>(
      `SELECT id, content_embedding FROM memori_entity_fact WHERE entity_id = $1 ORDER BY date_last_time DESC, num_times DESC, id DESC LIMIT ${limit}`,
      [entityId]
    );
    return results
      .filter(
        (r): r is { id: number | string; content_embedding: Buffer } =>
          r.content_embedding != null && r.content_embedding.length > 0
      )
      .map((r) => {
        return {
          id: r.id,
          content_embedding: bufferToFloat32Array(r.content_embedding),
        };
      });
  }

  public async getFactsByIds(factIds: (string | number)[]): Promise<CandidateFactRow[]> {
    if (factIds.length === 0) return [];
    const placeholders = factIds.map((_, i) => `$${i + 1}`).join(',');

    const factRows = await this.conn.execute<{
      id: number | string;
      content: string;
      date_created: string | Date;
    }>(
      `SELECT id, content, date_created FROM memori_entity_fact WHERE id IN (${placeholders})`,
      factIds
    );

    if (factRows.length === 0) return [];

    const factsById = new Map<string, CandidateFactRow>();
    const facts: CandidateFactRow[] = [];

    for (const row of factRows) {
      const fact: CandidateFactRow = {
        id: row.id,
        content: row.content,
        date_created: row.date_created ? new Date(row.date_created).toISOString() : '',
        summaries: [],
      };
      facts.push(fact);
      factsById.set(String(row.id), fact);
    }

    const summaryRows = await this.conn.execute<{
      fact_id: number | string;
      content: string;
      date_created: string | Date;
    }>(
      `SELECT m.fact_id, c.summary AS content, COALESCE(c.date_updated, c.date_created) AS date_created FROM memori_entity_fact_mention m JOIN memori_conversation c ON c.id = m.conversation_id WHERE m.fact_id IN (${placeholders}) AND c.summary IS NOT NULL AND c.summary <> ''`,
      factIds
    );

    for (const row of summaryRows) {
      const fact = factsById.get(String(row.fact_id));
      if (fact) {
        (fact.summaries ??= []).push({
          content: row.content,
          date_created: row.date_created ? new Date(row.date_created).toISOString() : '',
        });
      }
    }
    return facts;
  }
}

class KnowledgeGraph {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(
    entityId: number | string,
    semanticTriples: SemanticTriplePayload[]
  ): Promise<this> {
    if (semanticTriples.length === 0) return this;
    for (const triple of semanticTriples) {
      const subjName = typeof triple.subject === 'object' ? triple.subject.name : triple.subject;
      const subjType = typeof triple.subject === 'object' ? triple.subject.type : 'entity';
      const pred = triple.predicate;
      const objName = typeof triple.object === 'object' ? triple.object.name : triple.object;
      const objType = typeof triple.object === 'object' ? triple.object.type : 'entity';

      await this.conn.execute(
        `INSERT INTO memori_subject(uuid, name, type, uniq) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING`,
        [randomUUID(), subjName, subjType, generateUniq([subjName, subjType])]
      );
      const subjRes = await this.conn.execute<{ id: number | string }>(
        `SELECT id FROM memori_subject WHERE uniq = $1`,
        [generateUniq([subjName, subjType])]
      );

      await this.conn.execute(
        `INSERT INTO memori_predicate(uuid, content, uniq) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
        [randomUUID(), pred, generateUniq([pred])]
      );
      const predRes = await this.conn.execute<{ id: number | string }>(
        `SELECT id FROM memori_predicate WHERE uniq = $1`,
        [generateUniq([pred])]
      );

      await this.conn.execute(
        `INSERT INTO memori_object(uuid, name, type, uniq) VALUES ($1, $2, $3, $4) ON CONFLICT DO NOTHING`,
        [randomUUID(), objName, objType, generateUniq([objName, objType])]
      );
      const objRes = await this.conn.execute<{ id: number | string }>(
        `SELECT id FROM memori_object WHERE uniq = $1`,
        [generateUniq([objName, objType])]
      );

      if (entityId && subjRes.length > 0 && predRes.length > 0 && objRes.length > 0) {
        await this.conn.execute(
          `INSERT INTO memori_knowledge_graph(uuid, entity_id, subject_id, predicate_id, object_id, num_times, date_last_time) VALUES ($1, $2, $3, $4, $5, 1, CURRENT_TIMESTAMP) ON CONFLICT (entity_id, subject_id, predicate_id, object_id) DO UPDATE SET num_times = memori_knowledge_graph.num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
          [randomUUID(), entityId, subjRes[0].id, predRes[0].id, objRes[0].id]
        );
      }
    }
    return this;
  }
}

class Process {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(externalId: string | number): Promise<number | string | null> {
    await this.conn.execute(
      `INSERT INTO memori_process(uuid, external_id) VALUES ($1, $2) ON CONFLICT DO NOTHING`,
      [randomUUID(), externalId]
    );
    const res = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_process WHERE external_id = $1`,
      [externalId]
    );
    return res.length > 0 ? res[0].id : null;
  }
}

class ProcessAttribute {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(processId: number | string, attributes: string[]): Promise<this> {
    if (attributes.length === 0) return this;
    for (const attribute of attributes) {
      await this.conn.execute(
        `INSERT INTO memori_process_attribute(uuid, process_id, content, num_times, date_last_time, uniq) VALUES ($1, $2, $3, 1, CURRENT_TIMESTAMP, $4) ON CONFLICT (process_id, uniq) DO UPDATE SET num_times = memori_process_attribute.num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
        [randomUUID(), processId, attribute, generateUniq([attribute])]
      );
    }
    return this;
  }
}

class Session {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(
    uuid: string | number | null,
    entityId: number | string | null,
    processId: number | string | null
  ): Promise<number | string | null> {
    await this.conn.execute(
      `INSERT INTO memori_session(uuid, entity_id, process_id) VALUES ($1, $2, $3) ON CONFLICT DO NOTHING`,
      [uuid, entityId, processId]
    );
    const res = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_session WHERE uuid = $1`,
      [uuid]
    );
    return res.length > 0 ? res[0].id : null;
  }
}

class SchemaVersion {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(num: number): Promise<void> {
    await this.conn.execute(`INSERT INTO memori_schema_version(num) VALUES ($1)`, [num]);
  }
  public async delete(): Promise<void> {
    await this.conn.execute(`DELETE FROM memori_schema_version`);
  }
  public async read(): Promise<number | null> {
    try {
      const res = await this.conn.execute<{ num: number | string }>(
        `SELECT num FROM memori_schema_version`
      );
      return res.length > 0 ? Number(res[0].num) : null;
    } catch {
      return null;
    }
  }
}

class Schema {
  public readonly version: SchemaVersion;
  constructor(conn: StorageAdapter) {
    this.version = new SchemaVersion(conn);
  }
}

export class PostgresDriver extends BaseDriver {
  public readonly requiresRollbackOnError = true;
  public readonly migrations = postgresqlMigrations;
  constructor(conn: StorageAdapter) {
    super(conn);
    this.conversationMessage = new ConversationMessage(conn);
    this.conversationMessages = new ConversationMessages(conn);
    this.conversation = new Conversation(
      conn,
      this.conversationMessage as ConversationMessage,
      this.conversationMessages as ConversationMessages
    );
    this.entity = new Entity(conn);
    this.entityFact = new EntityFact(conn);
    this.knowledgeGraph = new KnowledgeGraph(conn);
    this.process = new Process(conn);
    this.processAttribute = new ProcessAttribute(conn);
    this.schema = new Schema(conn);
    this.session = new Session(conn);
  }
}

Registry.registerDriver('postgresql', PostgresDriver);
Registry.registerDriver('cockroachdb', PostgresDriver);
