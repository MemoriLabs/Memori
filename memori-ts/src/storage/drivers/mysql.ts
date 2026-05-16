import { randomUUID, createHash } from 'node:crypto';
import { StorageAdapter, BaseDriver } from '../base.js';
import { mysqlMigrations } from '../migrations/mysql.js';
import { Registry } from '../registry.js';
import {
  CandidateFactRow,
  ConversationSummaryRow,
  DetailedConversationMessageRow,
  SemanticTriplePayload,
} from '../../types/storage.js';
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
    content: string,
    options?: {
      trace?: string | null;
      source?: string | null;
      signal?: string | null;
    }
  ): Promise<void> {
    await this.conn.execute(
      `INSERT INTO memori_conversation_message(uuid, conversation_id, role, type, content, trace, source, signal) VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        randomUUID(),
        conversationId,
        role,
        type,
        content,
        options?.trace ?? null,
        options?.source ?? null,
        options?.signal ?? null,
      ]
    );
  }
}

class ConversationMessages {
  constructor(private readonly conn: StorageAdapter) {}
  public async read(
    conversationId: number | string
  ): Promise<Array<{ role: string; content: string }>> {
    const results = await this.conn.execute<{ role: string; content: string }>(
      `SELECT role, content FROM memori_conversation_message WHERE conversation_id = ? ORDER BY id`,
      [conversationId]
    );
    return results.map((row) => ({ content: row.content, role: row.role }));
  }

  public async readDetailed(
    conversationId: number | string
  ): Promise<DetailedConversationMessageRow[]> {
    return await this.conn.execute<DetailedConversationMessageRow>(
      `SELECT role, type, content, trace, source, signal, date_created FROM memori_conversation_message WHERE conversation_id = ? ORDER BY id`,
      [conversationId]
    );
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
    timeoutMinutes: number,
    projectId?: string | null
  ): Promise<number | null> {
    const existing = await this.conn.execute<{
      id: number | string;
      project_id: string | null;
      last_activity: string;
    }>(
      `SELECT c.id, c.project_id, COALESCE(MAX(m.date_created), c.date_created) as last_activity FROM memori_conversation c LEFT JOIN memori_conversation_message m ON m.conversation_id = c.id WHERE c.session_id = ? GROUP BY c.id, c.project_id, c.date_created`,
      [sessionId]
    );
    if (existing.length > 0) {
      const result = await this.conn.execute<{ minutes_since_activity: number }>(
        `SELECT TIMESTAMPDIFF(MINUTE, ?, CURRENT_TIMESTAMP) as minutes_since_activity`,
        [existing[0].last_activity]
      );
      if (result.length > 0 && result[0].minutes_since_activity <= timeoutMinutes) {
        if (projectId && existing[0].project_id !== projectId) {
          await this.conn.execute(`UPDATE memori_conversation SET project_id = ? WHERE id = ?`, [
            projectId,
            existing[0].id,
          ]);
        }
        return Number(existing[0].id);
      }
    }
    const uuid = randomUUID();
    await this.conn.execute(
      `INSERT IGNORE INTO memori_conversation(uuid, session_id, project_id) VALUES (?, ?, ?)`,
      [uuid, sessionId, projectId ?? null]
    );
    const newConv = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_conversation WHERE session_id = ?`,
      [sessionId]
    );
    return newConv.length > 0 ? Number(newConv[0].id) : null;
  }
  public async update(id: number | string, summary: string): Promise<this> {
    if (!summary) return this;
    await this.conn.execute(`UPDATE memori_conversation SET summary = ? WHERE id = ?`, [
      summary,
      id,
    ]);
    return this;
  }

  public async read(id: number | string): Promise<Record<string, unknown> | null> {
    const rows = await this.conn.execute(
      `SELECT id, uuid, session_id, project_id, summary, date_created, date_updated FROM memori_conversation WHERE id = ?`,
      [id]
    );
    return rows.length > 0 ? rows[0] : null;
  }

  public async searchSummaries(
    entityId: number | string,
    filters: {
      projectId?: string | null;
      sessionId?: string | null;
      dateStart?: string | null;
      dateEnd?: string | null;
      limit?: number;
    } = {}
  ): Promise<ConversationSummaryRow[]> {
    const clauses = ['s.entity_id = ?', `c.summary IS NOT NULL`, `c.summary <> ''`];
    const params: Array<string | number> = [entityId];

    if (filters.projectId) {
      clauses.push('c.project_id = ?');
      params.push(filters.projectId);
    }
    if (filters.sessionId) {
      clauses.push('s.uuid = ?');
      params.push(filters.sessionId);
    }
    if (filters.dateStart) {
      clauses.push(`COALESCE(c.date_updated, c.date_created) >= ?`);
      params.push(filters.dateStart);
    }
    if (filters.dateEnd) {
      clauses.push(`COALESCE(c.date_updated, c.date_created) <= ?`);
      params.push(filters.dateEnd);
    }
    params.push(filters.limit ?? 20);

    return await this.conn.execute<ConversationSummaryRow>(
      `SELECT c.id AS conversation_id, c.project_id AS project_id, s.uuid AS session_id, c.summary AS content, COALESCE(c.date_updated, c.date_created) AS date_created
         FROM memori_conversation c
         JOIN memori_session s ON s.id = c.session_id
        WHERE ${clauses.join(' AND ')}
        ORDER BY COALESCE(c.date_updated, c.date_created) DESC
        LIMIT ?`,
      params
    );
  }
}

class Entity {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(externalId: string | number): Promise<number | null> {
    await this.conn.execute(`INSERT IGNORE INTO memori_entity(uuid, external_id) VALUES (?, ?)`, [
      randomUUID(),
      externalId,
    ]);
    const res = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_entity WHERE external_id = ?`,
      [externalId]
    );
    return res.length > 0 ? Number(res[0].id) : null;
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
    for (let i = 0; i < facts.length; i++) {
      const fact = facts[i];
      const embedding =
        factEmbeddings && i < factEmbeddings.length ? factEmbeddings[i] : new Float32Array(0);
      if (embedding.length === 0) {
        continue;
      }
      if (embedding.length === 0) continue;
      const embeddingFormatted = formatEmbeddingForDb(embedding);
      const uniq = generateUniq([fact]);

      await this.conn.execute(
        `INSERT INTO memori_entity_fact(uuid, entity_id, content, content_embedding, num_times, date_last_time, uniq) VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?) ON DUPLICATE KEY UPDATE num_times = num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
        [randomUUID(), entityId, fact, embeddingFormatted, uniq]
      );
      if (conversationId) {
        const factRow = await this.conn.execute<{ id: number | string }>(
          `SELECT id FROM memori_entity_fact WHERE entity_id = ? AND uniq = ?`,
          [entityId, uniq]
        );
        if (factRow.length > 0) {
          await this.conn.execute(
            `INSERT IGNORE INTO memori_entity_fact_mention(uuid, entity_id, fact_id, conversation_id) VALUES (?, ?, ?, ?)`,
            [randomUUID(), entityId, factRow[0].id, conversationId]
          );
        }
      }
    }
    return this;
  }

  public async createWithoutEmbedding(entityId: number | string, content: string): Promise<void> {
    const uniq = generateUniq([content]);
    await this.conn.execute(
      `INSERT INTO memori_entity_fact(uuid, entity_id, content, content_embedding, num_times, date_last_time, uniq) VALUES (?, ?, ?, ?, 1, CURRENT_TIMESTAMP, ?) ON DUPLICATE KEY UPDATE num_times = num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
      [randomUUID(), entityId, content, Buffer.alloc(0), uniq]
    );
  }

  public async getEmbeddings(entityId: string | number, limit: number = 1000) {
    const results = await this.conn.execute<{
      id: number | string;
      content_embedding: Buffer | null;
    }>(
      `SELECT id, content_embedding FROM memori_entity_fact WHERE entity_id = ? ORDER BY date_last_time DESC, num_times DESC, id DESC LIMIT ${limit}`,
      [entityId]
    );
    return results
      .filter(
        (r): r is { id: number | string; content_embedding: Buffer } =>
          r.content_embedding != null && r.content_embedding.length > 0
      )
      .map((r) => {
        return {
          id: Number(r.id),
          content_embedding: bufferToFloat32Array(r.content_embedding),
        };
      });
  }

  public async getFactsByIds(factIds: (string | number)[]): Promise<CandidateFactRow[]> {
    if (factIds.length === 0) return [];
    const placeholders = factIds.map(() => '?').join(',');
    const factRows = await this.conn.execute<{
      id: number | string;
      content: string;
      date_created: string | Date;
    }>(
      `SELECT id, content, date_created FROM memori_entity_fact WHERE id IN (${placeholders})`,
      factIds
    );
    if (factRows.length === 0) return [];

    const factsById = new Map<number, CandidateFactRow>();
    const facts: CandidateFactRow[] = [];

    for (const row of factRows) {
      const numId = Number(row.id);
      const fact: CandidateFactRow = {
        id: numId,
        content: row.content,
        date_created: row.date_created ? new Date(row.date_created).toISOString() : '',
        summaries: [],
      };
      facts.push(fact);
      factsById.set(numId, fact);
    }

    const summaryRows = await this.conn.execute<{
      fact_id: number | string;
      content: string;
      date_created: string | Date;
      project_id: string | null;
      session_id: string | null;
      conversation_id: number | string;
    }>(
      `SELECT m.fact_id, c.summary AS content, COALESCE(c.date_updated, c.date_created) AS date_created, c.project_id AS project_id, s.uuid AS session_id, c.id AS conversation_id FROM memori_entity_fact_mention m JOIN memori_conversation c ON c.id = m.conversation_id JOIN memori_session s ON s.id = c.session_id WHERE m.fact_id IN (${placeholders}) AND c.summary IS NOT NULL AND c.summary <> ''`,
      factIds
    );
    for (const row of summaryRows) {
      const fact = factsById.get(Number(row.fact_id));
      if (fact) {
        (fact.summaries ??= []).push({
          content: row.content,
          date_created: row.date_created ? new Date(row.date_created).toISOString() : '',
          project_id: row.project_id ?? null,
          session_id: row.session_id ?? null,
          conversation_id: row.conversation_id,
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
        `INSERT IGNORE INTO memori_subject(uuid, name, type, uniq) VALUES (?, ?, ?, ?)`,
        [randomUUID(), subjName, subjType, generateUniq([subjName, subjType])]
      );
      const subjRes = await this.conn.execute<{ id: number | string }>(
        `SELECT id FROM memori_subject WHERE uniq = ?`,
        [generateUniq([subjName, subjType])]
      );

      await this.conn.execute(
        `INSERT IGNORE INTO memori_predicate(uuid, content, uniq) VALUES (?, ?, ?)`,
        [randomUUID(), pred, generateUniq([pred])]
      );
      const predRes = await this.conn.execute<{ id: number | string }>(
        `SELECT id FROM memori_predicate WHERE uniq = ?`,
        [generateUniq([pred])]
      );

      await this.conn.execute(
        `INSERT IGNORE INTO memori_object(uuid, name, type, uniq) VALUES (?, ?, ?, ?)`,
        [randomUUID(), objName, objType, generateUniq([objName, objType])]
      );
      const objRes = await this.conn.execute<{ id: number | string }>(
        `SELECT id FROM memori_object WHERE uniq = ?`,
        [generateUniq([objName, objType])]
      );

      if (entityId && subjRes.length > 0 && predRes.length > 0 && objRes.length > 0) {
        await this.conn.execute(
          `INSERT INTO memori_knowledge_graph(uuid, entity_id, subject_id, predicate_id, object_id, num_times, date_last_time) VALUES (?, ?, ?, ?, ?, 1, CURRENT_TIMESTAMP) ON DUPLICATE KEY UPDATE num_times = num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
          [randomUUID(), entityId, subjRes[0].id, predRes[0].id, objRes[0].id]
        );
      }
    }
    return this;
  }
}

class Process {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(externalId: string | number): Promise<number | null> {
    await this.conn.execute(`INSERT IGNORE INTO memori_process(uuid, external_id) VALUES (?, ?)`, [
      randomUUID(),
      externalId,
    ]);
    const res = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_process WHERE external_id = ?`,
      [externalId]
    );
    return res.length > 0 ? Number(res[0].id) : null;
  }
}

class ProcessAttribute {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(processId: number | string, attributes: string[]): Promise<this> {
    if (attributes.length === 0) return this;
    for (const attribute of attributes) {
      await this.conn.execute(
        `INSERT INTO memori_process_attribute(uuid, process_id, content, num_times, date_last_time, uniq) VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP, ?) ON DUPLICATE KEY UPDATE num_times = num_times + 1, date_last_time = CURRENT_TIMESTAMP`,
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
  ): Promise<number | null> {
    await this.conn.execute(
      `INSERT IGNORE INTO memori_session(uuid, entity_id, process_id) VALUES (?, ?, ?)`,
      [uuid, entityId, processId]
    );
    const res = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_session WHERE uuid = ?`,
      [uuid]
    );
    return res.length > 0 ? Number(res[0].id) : null;
  }

  public async read(uuid: string | number | null): Promise<number | null> {
    const res = await this.conn.execute<{ id: number | string }>(
      `SELECT id FROM memori_session WHERE uuid = ?`,
      [uuid]
    );
    return res.length > 0 ? Number(res[0].id) : null;
  }
}

class SchemaVersion {
  constructor(private readonly conn: StorageAdapter) {}
  public async create(num: number): Promise<void> {
    await this.conn.execute(`INSERT INTO memori_schema_version(num) VALUES (?)`, [num]);
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

export class MysqlDriver extends BaseDriver {
  public readonly requiresRollbackOnError = true;
  public readonly migrations = mysqlMigrations;
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

Registry.registerDriver('mysql', MysqlDriver);
