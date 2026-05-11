import { describe, it, expect, vi } from 'vitest';
import { SqliteDriver } from '../../src/storage/drivers/sqlite.js';
import { MysqlDriver } from '../../src/storage/drivers/mysql.js';
import { PostgresDriver } from '../../src/storage/drivers/postgresql.js';
import type { StorageAdapter } from '../../src/storage/base.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Returns a mock StorageAdapter whose execute() yields `responses` in order. */
function makeAdapter(
  responses: unknown[] = []
): StorageAdapter & { execute: ReturnType<typeof vi.fn> } {
  const queue = [...responses];
  return {
    execute: vi.fn().mockImplementation(() => queue.shift() ?? []),
    begin: vi.fn(),
    commit: vi.fn(),
    rollback: vi.fn(),
    close: vi.fn(),
    getDialect: vi.fn().mockReturnValue('sqlite'),
  };
}

/** Returns a mock StorageAdapter whose execute() resolves `responses` in order. */
function makeAsyncAdapter(
  responses: unknown[] = []
): StorageAdapter & { execute: ReturnType<typeof vi.fn> } {
  const queue = [...responses];
  return {
    execute: vi.fn().mockImplementation(async () => queue.shift() ?? []),
    begin: vi.fn().mockResolvedValue(undefined),
    commit: vi.fn().mockResolvedValue(undefined),
    rollback: vi.fn().mockResolvedValue(undefined),
    close: vi.fn().mockResolvedValue(undefined),
    getDialect: vi.fn().mockReturnValue('postgresql'),
  };
}

/** Creates a Float32Array-backed Buffer suitable for faking a stored embedding. */
function makeEmbeddingBuffer(values: number[]): Buffer {
  const arr = new Float32Array(values);
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength);
}

// ===========================================================================
// SqliteDriver
// ===========================================================================

describe('SqliteDriver', () => {
  it('exposes requiresRollbackOnError = false', () => {
    const driver = new SqliteDriver(makeAdapter());
    expect(driver.requiresRollbackOnError).toBe(false);
  });

  // ── ConversationMessage ──────────────────────────────────────────────────

  describe('conversationMessage.create()', () => {
    it('inserts a conversation message row', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      await driver.conversationMessage.create(1, 'user', 'text', 'hello');
      expect(conn.execute).toHaveBeenCalledOnce();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(
        /INSERT INTO memori_conversation_message/
      );
    });
  });

  // ── ConversationMessages ─────────────────────────────────────────────────

  describe('conversationMessages.read()', () => {
    it('returns mapped role/content pairs', async () => {
      const rows = [
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello' },
      ];
      const conn = makeAdapter([rows]);
      const driver = new SqliteDriver(conn);
      const result = await driver.conversationMessages.read(1);
      expect(result).toEqual([
        { role: 'user', content: 'hi' },
        { role: 'assistant', content: 'hello' },
      ]);
    });

    it('returns [] when no messages exist', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      expect(await driver.conversationMessages.read(1)).toEqual([]);
    });
  });

  // ── Conversation ─────────────────────────────────────────────────────────

  describe('conversation.create()', () => {
    it('returns existing id when last activity is within timeout', async () => {
      const conn = makeAdapter([
        [{ id: 3, last_activity: '2026-05-07 10:00:00' }], // existing conv
        [{ minutes_since_activity: 5 }], // elapsed time
      ]);
      const driver = new SqliteDriver(conn);
      const id = await driver.conversation.create(42, 30);
      expect(id).toBe(3);
      expect(conn.execute).toHaveBeenCalledTimes(2);
    });

    it('creates a new conversation when activity has timed out', async () => {
      const conn = makeAdapter([
        [{ id: 3, last_activity: '2026-05-07 09:00:00' }], // existing conv
        [{ minutes_since_activity: 90 }], // elapsed > timeout
        [], // INSERT
        [{ id: 7 }], // SELECT new id
      ]);
      const driver = new SqliteDriver(conn);
      const id = await driver.conversation.create(42, 30);
      expect(id).toBe(7);
    });

    it('creates a new conversation when none exists', async () => {
      const conn = makeAdapter([
        [], // no existing
        [], // INSERT
        [{ id: 5 }], // SELECT new id
      ]);
      const driver = new SqliteDriver(conn);
      const id = await driver.conversation.create(99, 30);
      expect(id).toBe(5);
    });

    it('returns null when the INSERT produces no row', async () => {
      const conn = makeAdapter([[], [], []]);
      const driver = new SqliteDriver(conn);
      const id = await driver.conversation.create(99, 30);
      expect(id).toBeNull();
    });
  });

  describe('conversation.update()', () => {
    it('executes UPDATE when summary is provided', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      await driver.conversation.update(1, 'summary text');
      expect(conn.execute).toHaveBeenCalledOnce();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/UPDATE memori_conversation/);
    });

    it('skips UPDATE when summary is empty', async () => {
      const conn = makeAdapter();
      const driver = new SqliteDriver(conn);
      await driver.conversation.update(1, '');
      expect(conn.execute).not.toHaveBeenCalled();
    });
  });

  // ── Entity ───────────────────────────────────────────────────────────────

  describe('entity.create()', () => {
    it('inserts entity and returns the new id', async () => {
      const conn = makeAdapter([[], [{ id: 10 }]]);
      const driver = new SqliteDriver(conn);
      expect(await driver.entity.create('user-1')).toBe(10);
    });

    it('returns null when no row found after insert', async () => {
      const conn = makeAdapter([[], []]);
      const driver = new SqliteDriver(conn);
      expect(await driver.entity.create('user-1')).toBeNull();
    });
  });

  // ── EntityFact ───────────────────────────────────────────────────────────

  describe('entityFact.create()', () => {
    it('returns early and calls no execute when facts array is empty', async () => {
      const conn = makeAdapter();
      const driver = new SqliteDriver(conn);
      await driver.entityFact.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('skips facts that have no embedding', async () => {
      const conn = makeAdapter();
      const driver = new SqliteDriver(conn);
      // Pass facts but provide an empty Float32Array as the embedding
      await driver.entityFact.create(1, ['fact with no embedding'], [new Float32Array(0)]);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts facts with valid embeddings', async () => {
      const embedding = new Float32Array([1.0, 2.0, 3.0]);
      const conn = makeAdapter([[], []]); // INSERT fact + (no conversationId, so no SELECT/mention)
      const driver = new SqliteDriver(conn);
      await driver.entityFact.create(1, ['some fact'], [embedding]);
      expect(conn.execute).toHaveBeenCalledOnce();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/INSERT INTO memori_entity_fact/);
    });

    it('links fact to conversation when conversationId is provided', async () => {
      const embedding = new Float32Array([1.0, 2.0]);
      const conn = makeAdapter([
        [], // INSERT fact
        [{ id: 99 }], // SELECT fact id
        [], // INSERT mention
      ]);
      const driver = new SqliteDriver(conn);
      await driver.entityFact.create(1, ['linked fact'], [embedding], 55);
      expect(conn.execute).toHaveBeenCalledTimes(3);
      const calls = (conn.execute as any).mock.calls.map((c: any[]) => c[0] as string);
      expect(calls[2]).toMatch(/INSERT INTO memori_entity_fact_mention/);
    });

    it('skips mention insert when fact id lookup returns empty', async () => {
      const embedding = new Float32Array([1.0]);
      const conn = makeAdapter([[], []]); // INSERT fact, SELECT returns []
      const driver = new SqliteDriver(conn);
      await driver.entityFact.create(1, ['fact'], [embedding], 55);
      // Only 2 calls: INSERT fact + SELECT fact id — no mention INSERT
      expect(conn.execute).toHaveBeenCalledTimes(2);
    });
  });

  describe('entityFact.createWithoutEmbedding()', () => {
    it('inserts a fact with empty embedding buffer', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      await driver.entityFact.createWithoutEmbedding(1, 'raw fact');
      expect(conn.execute).toHaveBeenCalledOnce();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/INSERT INTO memori_entity_fact/);
    });
  });

  describe('entityFact.getEmbeddings()', () => {
    it('returns parsed Float32Arrays for rows with non-empty embeddings', async () => {
      const buf = makeEmbeddingBuffer([1.5, 2.5]);
      const conn = makeAdapter([[{ id: 1, content_embedding: buf }]]);
      const driver = new SqliteDriver(conn);
      const result = await driver.entityFact.getEmbeddings(1);
      expect(result).toHaveLength(1);
      expect(result[0].id).toBe(1);
      expect(result[0].content_embedding).toBeInstanceOf(Float32Array);
    });

    it('filters out null and empty-buffer embeddings', async () => {
      const conn = makeAdapter([
        [
          { id: 1, content_embedding: null },
          { id: 2, content_embedding: Buffer.alloc(0) },
          { id: 3, content_embedding: makeEmbeddingBuffer([0.5]) },
        ],
      ]);
      const driver = new SqliteDriver(conn);
      const result = await driver.entityFact.getEmbeddings(1);
      expect(result).toHaveLength(1);
      expect(result[0].id).toBe(3);
    });

    it('returns [] when no rows found', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      expect(await driver.entityFact.getEmbeddings(1)).toEqual([]);
    });
  });

  describe('entityFact.getFactsByIds()', () => {
    it('returns [] immediately for empty factIds', async () => {
      const conn = makeAdapter();
      const driver = new SqliteDriver(conn);
      expect(await driver.entityFact.getFactsByIds([])).toEqual([]);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('returns [] when fact query returns no rows', async () => {
      const conn = makeAdapter([[], []]);
      const driver = new SqliteDriver(conn);
      expect(await driver.entityFact.getFactsByIds([1])).toEqual([]);
    });

    it('returns facts with their summaries', async () => {
      const conn = makeAdapter([
        [{ id: 1, content: 'the fact', date_created: '2026-01-01T00:00:00Z' }],
        [{ fact_id: 1, content: 'the summary', date_created: '2026-01-02T00:00:00Z' }],
      ]);
      const driver = new SqliteDriver(conn);
      const facts = await driver.entityFact.getFactsByIds([1]);
      expect(facts).toHaveLength(1);
      expect(facts[0].content).toBe('the fact');
      expect(facts[0].summaries).toHaveLength(1);
      expect(facts[0].summaries![0].content).toBe('the summary');
    });

    it('leaves summaries empty when no summary rows are found', async () => {
      const conn = makeAdapter([
        [{ id: 1, content: 'the fact', date_created: '2026-01-01T00:00:00Z' }],
        [],
      ]);
      const driver = new SqliteDriver(conn);
      const facts = await driver.entityFact.getFactsByIds([1]);
      expect(facts[0].summaries).toEqual([]);
    });

    it('ignores summary rows whose fact_id does not match any loaded fact', async () => {
      const conn = makeAdapter([
        [{ id: 1, content: 'fact', date_created: '2026-01-01T00:00:00Z' }],
        [{ fact_id: 999, content: 'orphan summary', date_created: '2026-01-02T00:00:00Z' }],
      ]);
      const driver = new SqliteDriver(conn);
      const facts = await driver.entityFact.getFactsByIds([1]);
      expect(facts[0].summaries).toEqual([]);
    });
  });

  // ── KnowledgeGraph ───────────────────────────────────────────────────────

  describe('knowledgeGraph.create()', () => {
    it('returns early and calls no execute for empty triples', async () => {
      const conn = makeAdapter();
      const driver = new SqliteDriver(conn);
      await driver.knowledgeGraph.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts subject/predicate/object and links to knowledge graph', async () => {
      const conn = makeAdapter([
        [], // INSERT subject
        [{ id: 1 }], // SELECT subject id
        [], // INSERT predicate
        [{ id: 2 }], // SELECT predicate id
        [], // INSERT object
        [{ id: 3 }], // SELECT object id
        [], // INSERT knowledge_graph
      ]);
      const driver = new SqliteDriver(conn);
      await driver.knowledgeGraph.create(10, [
        { subject: 'Alice', predicate: 'knows', object: 'Bob' },
      ]);
      expect(conn.execute).toHaveBeenCalledTimes(7);
    });

    it('accepts typed subject/object objects', async () => {
      const conn = makeAdapter([[], [{ id: 1 }], [], [{ id: 2 }], [], [{ id: 3 }], []]);
      const driver = new SqliteDriver(conn);
      await driver.knowledgeGraph.create(10, [
        {
          subject: { name: 'Alice', type: 'person' },
          predicate: 'likes',
          object: { name: 'cats', type: 'animal' },
        },
      ]);
      expect(conn.execute).toHaveBeenCalledTimes(7);
    });

    it('skips the knowledge_graph insert when any id lookup returns empty', async () => {
      const conn = makeAdapter([
        [], // INSERT subject
        [], // SELECT subject id → not found
        [], // INSERT predicate
        [{ id: 2 }],
        [], // INSERT object
        [{ id: 3 }],
        // No knowledge_graph INSERT
      ]);
      const driver = new SqliteDriver(conn);
      await driver.knowledgeGraph.create(10, [{ subject: 'X', predicate: 'rel', object: 'Y' }]);
      expect(conn.execute).toHaveBeenCalledTimes(6);
    });
  });

  // ── Process ──────────────────────────────────────────────────────────────

  describe('process.create()', () => {
    it('inserts process and returns id', async () => {
      const conn = makeAdapter([[], [{ id: 5 }]]);
      const driver = new SqliteDriver(conn);
      expect(await driver.process.create('proc-1')).toBe(5);
    });

    it('returns null when row not found after insert', async () => {
      const conn = makeAdapter([[], []]);
      const driver = new SqliteDriver(conn);
      expect(await driver.process.create('proc-1')).toBeNull();
    });
  });

  // ── ProcessAttribute ─────────────────────────────────────────────────────

  describe('processAttribute.create()', () => {
    it('returns early and calls no execute for empty attributes', async () => {
      const conn = makeAdapter();
      const driver = new SqliteDriver(conn);
      await driver.processAttribute.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts one row per attribute', async () => {
      const conn = makeAdapter([[], [], []]);
      const driver = new SqliteDriver(conn);
      await driver.processAttribute.create(1, ['a', 'b', 'c']);
      expect(conn.execute).toHaveBeenCalledTimes(3);
      expect((conn.execute as any).mock.calls[0][0]).toMatch(
        /INSERT INTO memori_process_attribute/
      );
    });
  });

  // ── Session ──────────────────────────────────────────────────────────────

  describe('session.create()', () => {
    it('inserts session and returns id', async () => {
      const conn = makeAdapter([[], [{ id: 11 }]]);
      const driver = new SqliteDriver(conn);
      expect(await driver.session.create('uuid-1', 1, 2)).toBe(11);
    });

    it('returns null when session row not found after insert', async () => {
      const conn = makeAdapter([[], []]);
      const driver = new SqliteDriver(conn);
      expect(await driver.session.create('uuid-1', 1, 2)).toBeNull();
    });
  });

  // ── Schema.version ───────────────────────────────────────────────────────

  describe('schema.version', () => {
    it('create() inserts schema version row', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      await driver.schema.version.create(3);
      expect(conn.execute).toHaveBeenCalledOnce();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/INSERT INTO memori_schema_version/);
    });

    it('delete() removes all schema version rows', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      await driver.schema.version.delete();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/DELETE FROM memori_schema_version/);
    });

    it('read() returns the current version number', async () => {
      const conn = makeAdapter([[{ num: 7 }]]);
      const driver = new SqliteDriver(conn);
      expect(await driver.schema.version.read()).toBe(7);
    });

    it('read() returns null when no version row exists', async () => {
      const conn = makeAdapter([[]]);
      const driver = new SqliteDriver(conn);
      expect(await driver.schema.version.read()).toBeNull();
    });
  });
});

// ===========================================================================
// MysqlDriver
// ===========================================================================

describe('MysqlDriver', () => {
  it('exposes requiresRollbackOnError = true', () => {
    const driver = new MysqlDriver(makeAsyncAdapter());
    expect(driver.requiresRollbackOnError).toBe(true);
  });

  // ── ConversationMessage ──────────────────────────────────────────────────

  describe('conversationMessage.create()', () => {
    it('inserts a conversation message row', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new MysqlDriver(conn);
      await driver.conversationMessage.create(1, 'user', null, 'hi');
      expect(conn.execute).toHaveBeenCalledOnce();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(
        /INSERT INTO memori_conversation_message/
      );
    });
  });

  // ── ConversationMessages ─────────────────────────────────────────────────

  describe('conversationMessages.read()', () => {
    it('maps rows to role/content pairs', async () => {
      const conn = makeAsyncAdapter([[{ role: 'assistant', content: 'hey' }]]);
      const driver = new MysqlDriver(conn);
      expect(await driver.conversationMessages.read(1)).toEqual([
        { role: 'assistant', content: 'hey' },
      ]);
    });
  });

  // ── Conversation ─────────────────────────────────────────────────────────

  describe('conversation.create()', () => {
    it('returns existing id within timeout', async () => {
      const conn = makeAsyncAdapter([
        [{ id: 3, last_activity: '2026-05-07 10:00:00' }],
        [{ minutes_since_activity: 2 }],
      ]);
      const driver = new MysqlDriver(conn);
      expect(await driver.conversation.create(42, 30)).toBe(3);
    });

    it('creates new conversation when none exists', async () => {
      const conn = makeAsyncAdapter([[], [], [{ id: 8 }]]);
      const driver = new MysqlDriver(conn);
      expect(await driver.conversation.create(99, 30)).toBe(8);
    });

    it('creates new conversation when last activity exceeded timeout', async () => {
      const conn = makeAsyncAdapter([
        [{ id: 3, last_activity: '2026-05-07 08:00:00' }],
        [{ minutes_since_activity: 200 }],
        [],
        [{ id: 9 }],
      ]);
      const driver = new MysqlDriver(conn);
      expect(await driver.conversation.create(42, 30)).toBe(9);
    });
  });

  describe('conversation.update()', () => {
    it('executes UPDATE when summary is non-empty', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new MysqlDriver(conn);
      await driver.conversation.update(1, 'a summary');
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/UPDATE memori_conversation/);
    });

    it('skips UPDATE when summary is empty', async () => {
      const conn = makeAsyncAdapter();
      const driver = new MysqlDriver(conn);
      await driver.conversation.update(1, '');
      expect(conn.execute).not.toHaveBeenCalled();
    });
  });

  // ── Entity ───────────────────────────────────────────────────────────────

  describe('entity.create()', () => {
    it('inserts entity and returns numeric id', async () => {
      const conn = makeAsyncAdapter([[], [{ id: '42' }]]);
      const driver = new MysqlDriver(conn);
      expect(await driver.entity.create('ext-1')).toBe(42);
    });

    it('returns null when lookup returns no rows', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new MysqlDriver(conn);
      expect(await driver.entity.create('ext-1')).toBeNull();
    });
  });

  // ── EntityFact ───────────────────────────────────────────────────────────

  describe('entityFact.create()', () => {
    it('returns early for empty facts array', async () => {
      const conn = makeAsyncAdapter();
      const driver = new MysqlDriver(conn);
      await driver.entityFact.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('skips facts with zero-length embeddings', async () => {
      const conn = makeAsyncAdapter();
      const driver = new MysqlDriver(conn);
      await driver.entityFact.create(1, ['no embedding'], [new Float32Array(0)]);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts fact with embedding and links to conversation', async () => {
      const embedding = new Float32Array([1.0, 2.0]);
      const conn = makeAsyncAdapter([[], [{ id: 7 }], []]);
      const driver = new MysqlDriver(conn);
      await driver.entityFact.create(1, ['a fact'], [embedding], 20);
      expect(conn.execute).toHaveBeenCalledTimes(3);
      const sqls = (conn.execute as any).mock.calls.map((c: any[]) => c[0] as string);
      expect(sqls[0]).toMatch(/INSERT INTO memori_entity_fact/);
      expect(sqls[2]).toMatch(/INSERT IGNORE INTO memori_entity_fact_mention/);
    });

    it('skips mention insert when fact lookup returns empty', async () => {
      const embedding = new Float32Array([1.0]);
      const conn = makeAsyncAdapter([[], []]);
      const driver = new MysqlDriver(conn);
      await driver.entityFact.create(1, ['fact'], [embedding], 20);
      // Only INSERT fact + SELECT fact id (no mention)
      expect(conn.execute).toHaveBeenCalledTimes(2);
    });
  });

  describe('entityFact.createWithoutEmbedding()', () => {
    it('inserts with empty embedding buffer', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new MysqlDriver(conn);
      await driver.entityFact.createWithoutEmbedding(1, 'raw fact');
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/INSERT INTO memori_entity_fact/);
    });
  });

  describe('entityFact.getEmbeddings()', () => {
    it('returns non-empty embeddings as Float32Arrays', async () => {
      const buf = makeEmbeddingBuffer([3.0]);
      const conn = makeAsyncAdapter([[{ id: 1, content_embedding: buf }]]);
      const driver = new MysqlDriver(conn);
      const result = await driver.entityFact.getEmbeddings(1);
      expect(result[0].id).toBe(1);
      expect(result[0].content_embedding).toBeInstanceOf(Float32Array);
    });

    it('filters null and empty-buffer embeddings', async () => {
      const conn = makeAsyncAdapter([
        [
          { id: 1, content_embedding: null },
          { id: 2, content_embedding: Buffer.alloc(0) },
        ],
      ]);
      const driver = new MysqlDriver(conn);
      expect(await driver.entityFact.getEmbeddings(1)).toEqual([]);
    });
  });

  describe('entityFact.getFactsByIds()', () => {
    it('returns [] for empty input without querying', async () => {
      const conn = makeAsyncAdapter();
      const driver = new MysqlDriver(conn);
      expect(await driver.entityFact.getFactsByIds([])).toEqual([]);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('returns [] when fact query returns no rows', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new MysqlDriver(conn);
      expect(await driver.entityFact.getFactsByIds([1])).toEqual([]);
    });

    it('returns facts with summaries attached', async () => {
      const conn = makeAsyncAdapter([
        [{ id: 1, content: 'fact', date_created: '2026-01-01T00:00:00Z' }],
        [{ fact_id: 1, content: 'sum', date_created: '2026-01-02T00:00:00Z' }],
      ]);
      const driver = new MysqlDriver(conn);
      const facts = await driver.entityFact.getFactsByIds([1]);
      expect(facts[0].content).toBe('fact');
      expect(facts[0].summaries![0].content).toBe('sum');
    });
  });

  // ── KnowledgeGraph ───────────────────────────────────────────────────────

  describe('knowledgeGraph.create()', () => {
    it('returns early for empty triples', async () => {
      const conn = makeAsyncAdapter();
      const driver = new MysqlDriver(conn);
      await driver.knowledgeGraph.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts subject/predicate/object and knowledge graph row', async () => {
      const conn = makeAsyncAdapter([[], [{ id: 1 }], [], [{ id: 2 }], [], [{ id: 3 }], []]);
      const driver = new MysqlDriver(conn);
      await driver.knowledgeGraph.create(5, [{ subject: 'A', predicate: 'rel', object: 'B' }]);
      expect(conn.execute).toHaveBeenCalledTimes(7);
    });

    it('skips knowledge_graph insert when subject lookup fails', async () => {
      const conn = makeAsyncAdapter([
        [],
        [], // INSERT + SELECT subject (empty)
        [],
        [{ id: 2 }], // INSERT + SELECT predicate
        [],
        [{ id: 3 }], // INSERT + SELECT object
      ]);
      const driver = new MysqlDriver(conn);
      await driver.knowledgeGraph.create(5, [{ subject: 'A', predicate: 'rel', object: 'B' }]);
      expect(conn.execute).toHaveBeenCalledTimes(6);
    });
  });

  // ── Process ──────────────────────────────────────────────────────────────

  describe('process.create()', () => {
    it('inserts and returns numeric id', async () => {
      const conn = makeAsyncAdapter([[], [{ id: '5' }]]);
      const driver = new MysqlDriver(conn);
      expect(await driver.process.create('p1')).toBe(5);
    });

    it('returns null when lookup returns no rows', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new MysqlDriver(conn);
      expect(await driver.process.create('p1')).toBeNull();
    });
  });

  // ── ProcessAttribute ─────────────────────────────────────────────────────

  describe('processAttribute.create()', () => {
    it('returns early for empty attributes', async () => {
      const conn = makeAsyncAdapter();
      const driver = new MysqlDriver(conn);
      await driver.processAttribute.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts one row per attribute', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new MysqlDriver(conn);
      await driver.processAttribute.create(1, ['x', 'y']);
      expect(conn.execute).toHaveBeenCalledTimes(2);
    });
  });

  // ── Session ──────────────────────────────────────────────────────────────

  describe('session.create()', () => {
    it('inserts session and returns numeric id', async () => {
      const conn = makeAsyncAdapter([[], [{ id: '11' }]]);
      const driver = new MysqlDriver(conn);
      expect(await driver.session.create('uuid-1', 1, 2)).toBe(11);
    });

    it('returns null when session not found after insert', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new MysqlDriver(conn);
      expect(await driver.session.create('uuid-1', 1, 2)).toBeNull();
    });
  });

  // ── Schema.version ───────────────────────────────────────────────────────

  describe('schema.version', () => {
    it('create() inserts version row', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new MysqlDriver(conn);
      await driver.schema.version.create(2);
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/INSERT INTO memori_schema_version/);
    });

    it('delete() removes version rows', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new MysqlDriver(conn);
      await driver.schema.version.delete();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/DELETE FROM memori_schema_version/);
    });

    it('read() returns version number', async () => {
      const conn = makeAsyncAdapter([[{ num: '5' }]]);
      const driver = new MysqlDriver(conn);
      expect(await driver.schema.version.read()).toBe(5);
    });

    it('read() returns null when no version row exists', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new MysqlDriver(conn);
      expect(await driver.schema.version.read()).toBeNull();
    });

    it('read() returns null when execute throws', async () => {
      const conn = makeAsyncAdapter();
      (conn.execute as any).mockRejectedValue(new Error('table not found'));
      const driver = new MysqlDriver(conn);
      expect(await driver.schema.version.read()).toBeNull();
    });
  });
});

// ===========================================================================
// PostgresDriver
// ===========================================================================

describe('PostgresDriver', () => {
  it('exposes requiresRollbackOnError = true', () => {
    const driver = new PostgresDriver(makeAsyncAdapter());
    expect(driver.requiresRollbackOnError).toBe(true);
  });

  it('registers under both "postgresql" and "cockroachdb" (driver instantiates fine)', () => {
    // Both dialects use PostgresDriver — just confirm construction succeeds
    expect(() => new PostgresDriver(makeAsyncAdapter())).not.toThrow();
  });

  // ── ConversationMessage ──────────────────────────────────────────────────

  describe('conversationMessage.create()', () => {
    it('inserts a message row with $N placeholders', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new PostgresDriver(conn);
      await driver.conversationMessage.create(1, 'user', 'text', 'hello');
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/\$1/);
    });
  });

  // ── ConversationMessages ─────────────────────────────────────────────────

  describe('conversationMessages.read()', () => {
    it('returns mapped role/content pairs', async () => {
      const conn = makeAsyncAdapter([[{ role: 'user', content: 'q' }]]);
      const driver = new PostgresDriver(conn);
      expect(await driver.conversationMessages.read(1)).toEqual([{ role: 'user', content: 'q' }]);
    });
  });

  // ── Conversation ─────────────────────────────────────────────────────────

  describe('conversation.create()', () => {
    it('returns existing id within timeout', async () => {
      const conn = makeAsyncAdapter([
        [{ id: 4, last_activity: '2026-05-07T10:00:00' }],
        [{ minutes_since_activity: 1 }],
      ]);
      const driver = new PostgresDriver(conn);
      expect(await driver.conversation.create(1, 30)).toBe(4);
    });

    it('creates new conversation when none exists', async () => {
      const conn = makeAsyncAdapter([[], [], [{ id: 6 }]]);
      const driver = new PostgresDriver(conn);
      expect(await driver.conversation.create(1, 30)).toBe(6);
    });

    it('creates new conversation after timeout expiry', async () => {
      const conn = makeAsyncAdapter([
        [{ id: 4, last_activity: '2026-05-07T09:00:00' }],
        [{ minutes_since_activity: 120 }],
        [],
        [{ id: 10 }],
      ]);
      const driver = new PostgresDriver(conn);
      expect(await driver.conversation.create(1, 30)).toBe(10);
    });
  });

  describe('conversation.update()', () => {
    it('executes UPDATE when summary is non-empty', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new PostgresDriver(conn);
      await driver.conversation.update(1, 'summary');
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/UPDATE memori_conversation/);
    });

    it('skips UPDATE for empty summary', async () => {
      const conn = makeAsyncAdapter();
      const driver = new PostgresDriver(conn);
      await driver.conversation.update(1, '');
      expect(conn.execute).not.toHaveBeenCalled();
    });
  });

  // ── Entity ───────────────────────────────────────────────────────────────

  describe('entity.create()', () => {
    it('inserts entity and returns id', async () => {
      const conn = makeAsyncAdapter([[], [{ id: 'ent-1' }]]);
      const driver = new PostgresDriver(conn);
      expect(await driver.entity.create('ext-1')).toBe('ent-1');
    });

    it('returns null when row not found after insert', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new PostgresDriver(conn);
      expect(await driver.entity.create('ext-1')).toBeNull();
    });
  });

  // ── EntityFact ───────────────────────────────────────────────────────────

  describe('entityFact.create()', () => {
    it('returns early for empty facts array', async () => {
      const conn = makeAsyncAdapter();
      const driver = new PostgresDriver(conn);
      await driver.entityFact.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('skips facts with empty embeddings', async () => {
      const conn = makeAsyncAdapter();
      const driver = new PostgresDriver(conn);
      await driver.entityFact.create(1, ['no emb'], [new Float32Array(0)]);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('bulk-inserts facts with valid embeddings', async () => {
      const embedding = new Float32Array([1.0, 2.0]);
      const conn = makeAsyncAdapter([[]]);
      const driver = new PostgresDriver(conn);
      await driver.entityFact.create(1, ['fact'], [embedding]);
      expect(conn.execute).toHaveBeenCalledOnce();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/INSERT INTO memori_entity_fact/);
    });

    it('links facts to conversation when conversationId is provided', async () => {
      const embedding = new Float32Array([1.0]);
      const conn = makeAsyncAdapter([
        [], // bulk INSERT facts
        [{ id: 5 }], // SELECT inserted fact ids
        [], // bulk INSERT mentions
      ]);
      const driver = new PostgresDriver(conn);
      await driver.entityFact.create(1, ['fact'], [embedding], 99);
      expect(conn.execute).toHaveBeenCalledTimes(3);
      const sqls = (conn.execute as any).mock.calls.map((c: any[]) => c[0] as string);
      expect(sqls[2]).toMatch(/INSERT INTO memori_entity_fact_mention/);
    });

    it('skips mention insert when no fact ids returned', async () => {
      const embedding = new Float32Array([1.0]);
      const conn = makeAsyncAdapter([[], []]);
      const driver = new PostgresDriver(conn);
      await driver.entityFact.create(1, ['fact'], [embedding], 99);
      // bulk INSERT + SELECT returns [] → no mention insert
      expect(conn.execute).toHaveBeenCalledTimes(2);
    });
  });

  describe('entityFact.createWithoutEmbedding()', () => {
    it('inserts with empty embedding buffer using $N placeholders', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new PostgresDriver(conn);
      await driver.entityFact.createWithoutEmbedding(1, 'content');
      const sql = (conn.execute as any).mock.calls[0][0];
      expect(sql).toMatch(/INSERT INTO memori_entity_fact/);
      expect(sql).toMatch(/\$1/);
    });
  });

  describe('entityFact.getEmbeddings()', () => {
    it('returns Float32Array embeddings filtering out null/empty', async () => {
      const buf = makeEmbeddingBuffer([1.0, 2.0]);
      const conn = makeAsyncAdapter([
        [
          { id: 1, content_embedding: buf },
          { id: 2, content_embedding: null },
          { id: 3, content_embedding: Buffer.alloc(0) },
        ],
      ]);
      const driver = new PostgresDriver(conn);
      const result = await driver.entityFact.getEmbeddings(1);
      expect(result).toHaveLength(1);
      expect(result[0].content_embedding).toBeInstanceOf(Float32Array);
    });
  });

  describe('entityFact.getFactsByIds()', () => {
    it('returns [] for empty input without querying', async () => {
      const conn = makeAsyncAdapter();
      const driver = new PostgresDriver(conn);
      expect(await driver.entityFact.getFactsByIds([])).toEqual([]);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('returns [] when fact rows are empty', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new PostgresDriver(conn);
      expect(await driver.entityFact.getFactsByIds([1])).toEqual([]);
    });

    it('returns facts with summaries', async () => {
      const conn = makeAsyncAdapter([
        [{ id: '1', content: 'fact', date_created: '2026-01-01T00:00:00Z' }],
        [{ fact_id: '1', content: 'sum', date_created: '2026-01-02T00:00:00Z' }],
      ]);
      const driver = new PostgresDriver(conn);
      const facts = await driver.entityFact.getFactsByIds(['1']);
      expect(facts[0].content).toBe('fact');
      expect(facts[0].summaries![0].content).toBe('sum');
    });
  });

  // ── KnowledgeGraph ───────────────────────────────────────────────────────

  describe('knowledgeGraph.create()', () => {
    it('returns early for empty triples', async () => {
      const conn = makeAsyncAdapter();
      const driver = new PostgresDriver(conn);
      await driver.knowledgeGraph.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts subject/predicate/object and knowledge graph row', async () => {
      const conn = makeAsyncAdapter([[], [{ id: 1 }], [], [{ id: 2 }], [], [{ id: 3 }], []]);
      const driver = new PostgresDriver(conn);
      await driver.knowledgeGraph.create(5, [
        { subject: 'Alice', predicate: 'knows', object: 'Bob' },
      ]);
      expect(conn.execute).toHaveBeenCalledTimes(7);
    });

    it('skips knowledge_graph insert when any id lookup returns empty', async () => {
      const conn = makeAsyncAdapter([
        [],
        [], // subject
        [],
        [{ id: 2 }], // predicate
        [],
        [{ id: 3 }], // object
      ]);
      const driver = new PostgresDriver(conn);
      await driver.knowledgeGraph.create(5, [{ subject: 'A', predicate: 'rel', object: 'B' }]);
      expect(conn.execute).toHaveBeenCalledTimes(6);
    });
  });

  // ── Process ──────────────────────────────────────────────────────────────

  describe('process.create()', () => {
    it('inserts and returns id', async () => {
      const conn = makeAsyncAdapter([[], [{ id: 'p-1' }]]);
      const driver = new PostgresDriver(conn);
      expect(await driver.process.create('ext-p')).toBe('p-1');
    });

    it('returns null when lookup is empty', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new PostgresDriver(conn);
      expect(await driver.process.create('ext-p')).toBeNull();
    });
  });

  // ── ProcessAttribute ─────────────────────────────────────────────────────

  describe('processAttribute.create()', () => {
    it('returns early for empty attributes', async () => {
      const conn = makeAsyncAdapter();
      const driver = new PostgresDriver(conn);
      await driver.processAttribute.create(1, []);
      expect(conn.execute).not.toHaveBeenCalled();
    });

    it('inserts one row per attribute', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new PostgresDriver(conn);
      await driver.processAttribute.create(1, ['attr1', 'attr2']);
      expect(conn.execute).toHaveBeenCalledTimes(2);
    });
  });

  // ── Session ──────────────────────────────────────────────────────────────

  describe('session.create()', () => {
    it('inserts session and returns id', async () => {
      const conn = makeAsyncAdapter([[], [{ id: 'sess-1' }]]);
      const driver = new PostgresDriver(conn);
      expect(await driver.session.create('uuid-1', 1, 2)).toBe('sess-1');
    });

    it('returns null when session row not found', async () => {
      const conn = makeAsyncAdapter([[], []]);
      const driver = new PostgresDriver(conn);
      expect(await driver.session.create('uuid-1', 1, 2)).toBeNull();
    });
  });

  // ── Schema.version ───────────────────────────────────────────────────────

  describe('schema.version', () => {
    it('create() inserts version row', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new PostgresDriver(conn);
      await driver.schema.version.create(1);
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/INSERT INTO memori_schema_version/);
    });

    it('delete() removes version rows', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new PostgresDriver(conn);
      await driver.schema.version.delete();
      expect((conn.execute as any).mock.calls[0][0]).toMatch(/DELETE FROM memori_schema_version/);
    });

    it('read() returns version number', async () => {
      const conn = makeAsyncAdapter([[{ num: '3' }]]);
      const driver = new PostgresDriver(conn);
      expect(await driver.schema.version.read()).toBe(3);
    });

    it('read() returns null when no version row exists', async () => {
      const conn = makeAsyncAdapter([[]]);
      const driver = new PostgresDriver(conn);
      expect(await driver.schema.version.read()).toBeNull();
    });

    it('read() returns null when execute throws (table does not exist yet)', async () => {
      const conn = makeAsyncAdapter();
      (conn.execute as any).mockRejectedValue(new Error('relation not found'));
      const driver = new PostgresDriver(conn);
      expect(await driver.schema.version.read()).toBeNull();
    });
  });
});
