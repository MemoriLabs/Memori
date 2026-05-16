import { describe, it, expect, vi, beforeEach } from 'vitest';

// vi.mock is hoisted — all variables it references must be declared with vi.hoisted.
const {
  mockExecute,
  mockBegin,
  mockCommit,
  mockRollback,
  mockClose,
  mockGetDialect,
  mockDriverOps,
} = vi.hoisted(() => {
  const mockDriverOps = {
    session: {
      create: vi.fn().mockResolvedValue('sess-1'),
      read: vi.fn().mockResolvedValue('sess-1'),
    },
    conversation: {
      create: vi.fn().mockResolvedValue('conv-1'),
      update: vi.fn().mockResolvedValue(undefined),
      read: vi.fn().mockResolvedValue({ id: 'conv-1' }),
      searchSummaries: vi.fn().mockResolvedValue([]),
    },
    conversationMessage: { create: vi.fn().mockResolvedValue(undefined) },
    conversationMessages: {
      read: vi.fn().mockResolvedValue([]),
      readDetailed: vi.fn().mockResolvedValue([]),
    },
    entity: { create: vi.fn().mockResolvedValue('ent-1') },
    entityFact: {
      create: vi.fn().mockResolvedValue(undefined),
      createWithoutEmbedding: vi.fn().mockResolvedValue(undefined),
      getEmbeddings: vi.fn().mockResolvedValue([]),
      getFactsByIds: vi.fn().mockResolvedValue([]),
    },
    knowledgeGraph: { create: vi.fn().mockResolvedValue(undefined) },
    process: { create: vi.fn().mockResolvedValue('proc-1') },
    processAttribute: { create: vi.fn().mockResolvedValue(undefined) },
    schema: {
      version: {
        read: vi.fn().mockResolvedValue(0),
        delete: vi.fn().mockResolvedValue(undefined),
        create: vi.fn().mockResolvedValue(undefined),
      },
    },
  };

  return {
    mockExecute: vi.fn().mockResolvedValue([]),
    mockBegin: vi.fn().mockResolvedValue(undefined),
    mockCommit: vi.fn().mockResolvedValue(undefined),
    mockRollback: vi.fn().mockResolvedValue(undefined),
    mockClose: vi.fn().mockResolvedValue(undefined),
    mockGetDialect: vi.fn().mockReturnValue('sqlite'),
    mockDriverOps,
  };
});

vi.mock('../../src/storage/registry.js', () => ({
  Registry: {
    registerAdapter: vi.fn(),
    registerDriver: vi.fn(),
    getAdapter: vi.fn().mockReturnValue({
      execute: (...args: unknown[]) => mockExecute(...args),
      begin: (...args: unknown[]) => mockBegin(...args),
      commit: (...args: unknown[]) => mockCommit(...args),
      rollback: (...args: unknown[]) => mockRollback(...args),
      close: (...args: unknown[]) => mockClose(...args),
      getDialect: () => mockGetDialect(),
    }),
    getDriver: vi.fn().mockReturnValue({
      requiresRollbackOnError: false,
      migrations: {},
      session: {
        create: (...a: unknown[]) => mockDriverOps.session.create(...a),
        read: (...a: unknown[]) => mockDriverOps.session.read(...a),
      },
      conversation: {
        create: (...a: unknown[]) => mockDriverOps.conversation.create(...a),
        update: (...a: unknown[]) => mockDriverOps.conversation.update(...a),
        read: (...a: unknown[]) => mockDriverOps.conversation.read(...a),
        searchSummaries: (...a: unknown[]) => mockDriverOps.conversation.searchSummaries(...a),
      },
      conversationMessage: {
        create: (...a: unknown[]) => mockDriverOps.conversationMessage.create(...a),
      },
      conversationMessages: {
        read: (...a: unknown[]) => mockDriverOps.conversationMessages.read(...a),
        readDetailed: (...a: unknown[]) => mockDriverOps.conversationMessages.readDetailed(...a),
      },
      entity: { create: (...a: unknown[]) => mockDriverOps.entity.create(...a) },
      entityFact: {
        create: (...a: unknown[]) => mockDriverOps.entityFact.create(...a),
        createWithoutEmbedding: (...a: unknown[]) =>
          mockDriverOps.entityFact.createWithoutEmbedding(...a),
        getEmbeddings: (...a: unknown[]) => mockDriverOps.entityFact.getEmbeddings(...a),
        getFactsByIds: (...a: unknown[]) => mockDriverOps.entityFact.getFactsByIds(...a),
      },
      knowledgeGraph: { create: (...a: unknown[]) => mockDriverOps.knowledgeGraph.create(...a) },
      process: { create: (...a: unknown[]) => mockDriverOps.process.create(...a) },
      processAttribute: {
        create: (...a: unknown[]) => mockDriverOps.processAttribute.create(...a),
      },
      schema: {
        version: {
          read: (...a: unknown[]) => mockDriverOps.schema.version.read(...a),
          delete: (...a: unknown[]) => mockDriverOps.schema.version.delete(...a),
          create: (...a: unknown[]) => mockDriverOps.schema.version.create(...a),
        },
      },
    }),
  },
}));

import { StorageManager } from '../../src/storage/manager.js';
import { Registry } from '../../src/storage/registry.js';
import type { WriteBatch } from '../../src/types/storage.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const makeFactory = () => () => ({ __test: true });

function makeConversationMessageBatch(): WriteBatch {
  return {
    ops: [
      {
        op_type: 'conversation_message.create',
        payload: {
          conversation_id: 'conv-123',
          messages: [
            { role: 'user', content: 'hello' },
            { role: 'assistant', content: 'hi' },
          ],
        },
      },
    ],
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('StorageManager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockCommit.mockResolvedValue(undefined);
    mockRollback.mockResolvedValue(undefined);
    mockBegin.mockResolvedValue(undefined);
    mockDriverOps.session.create.mockResolvedValue('sess-1');
    mockDriverOps.session.read.mockResolvedValue('sess-1');
    mockDriverOps.conversation.create.mockResolvedValue('conv-1');
  });

  it('initialises adapter and driver from the factory', () => {
    const factory = makeFactory();
    new StorageManager(factory);
    expect(Registry.getAdapter).toHaveBeenCalledWith(factory);
  });

  it('getDialect() returns the adapter dialect', () => {
    const manager = new StorageManager(makeFactory());
    expect(manager.getDialect()).toBe('sqlite');
  });

  it('writeBatch() returns { written_ops: 0 } for empty ops', async () => {
    const manager = new StorageManager(makeFactory());
    expect(await manager.writeBatch({ ops: [] })).toEqual({ written_ops: 0 });
  });

  it('writeBatch() processes conversation_message.create and commits', async () => {
    const manager = new StorageManager(makeFactory());
    const result = await manager.writeBatch(makeConversationMessageBatch());
    expect(mockBegin).toHaveBeenCalled();
    expect(mockCommit).toHaveBeenCalled();
    expect(result.written_ops).toBe(1);
  });

  it('writeBatch() passes project_id and serialized trace through local message persistence', async () => {
    const manager = new StorageManager(makeFactory());
    await manager.writeBatch({
      ops: [
        {
          op_type: 'conversation_message.create',
          payload: {
            conversation_id: 'conv-123',
            project_id: 'proj-1',
            messages: [
              {
                role: 'assistant',
                content: 'done',
                type: 'text',
                trace: { tools: [] },
              },
            ],
          },
        },
      ],
    });

    expect(mockDriverOps.conversation.create).toHaveBeenCalledWith('sess-1', 30, 'proj-1');
    expect(mockDriverOps.conversationMessage.create).toHaveBeenCalledWith(
      'conv-1',
      'assistant',
      'text',
      'done',
      expect.objectContaining({ trace: '{"tools":[]}' })
    );
  });

  it('writeBatch() rolls back and returns 0 written_ops on driver error', async () => {
    mockDriverOps.session.create.mockRejectedValueOnce(new Error('DB error'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const manager = new StorageManager(makeFactory());
    const result = await manager.writeBatch(makeConversationMessageBatch());
    expect(mockRollback).toHaveBeenCalled();
    expect(result).toEqual({ written_ops: 0 });
    consoleSpy.mockRestore();
  });

  it('writeBatch() retries on error code 40001 and succeeds on second attempt', async () => {
    mockCommit
      .mockRejectedValueOnce(Object.assign(new Error('retry'), { code: '40001' }))
      .mockResolvedValue(undefined);

    vi.useFakeTimers();
    const manager = new StorageManager(makeFactory());
    const resultPromise = manager.writeBatch(makeConversationMessageBatch());
    await vi.runAllTimersAsync();
    const result = await resultPromise;

    expect(mockCommit).toHaveBeenCalledTimes(2);
    expect(result.written_ops).toBe(1);
    vi.useRealTimers();
  });

  it('writeBatch() gives up after max retries and logs the error', async () => {
    const serialErr = Object.assign(new Error('keep retrying'), { code: '40001' });
    mockCommit.mockRejectedValue(serialErr);
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    vi.useFakeTimers();
    const manager = new StorageManager(makeFactory());
    const resultPromise = manager.writeBatch(makeConversationMessageBatch());
    await vi.runAllTimersAsync();
    const result = await resultPromise;

    expect(result).toEqual({ written_ops: 0 });
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
    vi.useRealTimers();
  });

  it('writeBatch() processes entity_fact.create', async () => {
    const manager = new StorageManager(makeFactory());
    const batch: WriteBatch = {
      ops: [
        {
          op_type: 'entity_fact.create',
          payload: { entity_id: 'ent-1', facts: ['fact A'], conversation_id: 'conv-1' },
        },
      ],
    };
    const result = await manager.writeBatch(batch);
    expect(result.written_ops).toBe(1);
    expect(mockDriverOps.entityFact.create).toHaveBeenCalled();
  });

  it('writeBatch() processes knowledge_graph.create', async () => {
    const manager = new StorageManager(makeFactory());
    const batch: WriteBatch = {
      ops: [
        {
          op_type: 'knowledge_graph.create',
          payload: { entity_id: 'ent-1', semantic_triples: [] },
        },
      ],
    };
    const result = await manager.writeBatch(batch);
    expect(result.written_ops).toBe(1);
    expect(mockDriverOps.knowledgeGraph.create).toHaveBeenCalled();
  });

  it('writeBatch() processes process_attribute.create with array attributes', async () => {
    const manager = new StorageManager(makeFactory());
    const batch: WriteBatch = {
      ops: [
        {
          op_type: 'process_attribute.create',
          payload: { process_id: 'proc-1', attributes: ['attr1', 'attr2'] },
        },
      ],
    };
    const result = await manager.writeBatch(batch);
    expect(result.written_ops).toBe(1);
    expect(mockDriverOps.processAttribute.create).toHaveBeenCalledWith('proc-1', [
      'attr1',
      'attr2',
    ]);
  });

  it('writeBatch() processes process_attribute.create with object attributes', async () => {
    const manager = new StorageManager(makeFactory());
    const batch: WriteBatch = {
      ops: [
        {
          op_type: 'process_attribute.create',
          payload: { process_id: 'proc-1', attributes: { key: 'val' } },
        },
      ],
    };
    const result = await manager.writeBatch(batch);
    expect(result.written_ops).toBe(1);
    expect(mockDriverOps.processAttribute.create).toHaveBeenCalledWith('proc-1', ['val']);
  });

  it('writeBatch() processes conversation.update', async () => {
    const manager = new StorageManager(makeFactory());
    const batch: WriteBatch = {
      ops: [
        {
          op_type: 'conversation.update',
          payload: { conversation_id: 'conv-1', summary: 'session summary' },
        },
      ],
    };
    const result = await manager.writeBatch(batch);
    expect(result.written_ops).toBe(1);
    expect(mockDriverOps.conversation.update).toHaveBeenCalled();
  });

  it('writeBatch() processes upsert_fact', async () => {
    const manager = new StorageManager(makeFactory());
    const batch: WriteBatch = {
      ops: [
        {
          op_type: 'upsert_fact',
          payload: { entity_id: 'ent-1', content: 'User likes coffee' },
        },
      ],
    };
    const result = await manager.writeBatch(batch);
    expect(result.written_ops).toBe(1);
    expect(mockDriverOps.entityFact.createWithoutEmbedding).toHaveBeenCalledWith(
      'ent-1',
      'User likes coffee'
    );
  });

  it('writeBatch() logs error on unknown op_type and returns 0', async () => {
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const manager = new StorageManager(makeFactory());
    const result = await manager.writeBatch({
      ops: [{ op_type: '__unknown__', payload: {} } as any],
    });
    expect(result).toEqual({ written_ops: 0 });
    consoleSpy.mockRestore();
  });

  it('fetchEmbeddings() delegates to driver', async () => {
    mockDriverOps.entityFact.getEmbeddings.mockResolvedValue([
      { id: 1, content_embedding: new Float32Array(3) },
    ]);
    const manager = new StorageManager(makeFactory());
    const rows = await manager.fetchEmbeddings('ent-1', 10);
    expect(mockDriverOps.entityFact.getEmbeddings).toHaveBeenCalledWith('ent-1', 10);
    expect(rows).toHaveLength(1);
  });

  it('fetchFactsByIds() delegates to driver', async () => {
    mockDriverOps.entityFact.getFactsByIds.mockResolvedValue([{ id: 1, content: 'fact' }]);
    const manager = new StorageManager(makeFactory());
    const rows = await manager.fetchFactsByIds([1, 2]);
    expect(rows).toHaveLength(1);
  });

  it('getConversationHistoryDetailed() delegates to the detailed reader', async () => {
    mockDriverOps.conversationMessages.readDetailed.mockResolvedValue([
      { role: 'assistant', content: 'done', trace: '{"tools":[]}' },
    ]);

    const manager = new StorageManager(makeFactory());
    const rows = await manager.getConversationHistoryDetailed('sess-1');

    expect(rows).toEqual([{ role: 'assistant', content: 'done', trace: '{"tools":[]}' }]);
  });

  it('getConversationSummaries() resolves entity and delegates to scoped summary search', async () => {
    mockDriverOps.conversation.searchSummaries.mockResolvedValue([
      { content: 'summary', date_created: '2026-01-01T00:00:00Z', conversation_id: 'conv-1' },
    ]);

    const manager = new StorageManager(makeFactory());
    const rows = await manager.getConversationSummaries('entity-1', { projectId: 'proj-1' });

    expect(mockDriverOps.entity.create).toHaveBeenCalledWith('entity-1');
    expect(mockDriverOps.conversation.searchSummaries).toHaveBeenCalledWith('ent-1', {
      projectId: 'proj-1',
    });
    expect(rows).toHaveLength(1);
  });

  it('close() calls adapter.close()', async () => {
    const manager = new StorageManager(makeFactory());
    await manager.close();
    expect(mockClose).toHaveBeenCalled();
  });
});
