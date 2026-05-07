import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NativeEngine } from '../../src/core/engine.js';
import { MemoriEngine } from '../../src/native/index.js';

// setup.ts mocks src/native/index.js — MemoriEngine is a vi.fn() that returns a stub object.

const RETRIEVE_REQ = { entity_id: 'u-1', query_text: 'q', dense_limit: 5, limit: 3 };

function makeBridge() {
  return {
    fetchEmbeddings: vi.fn().mockResolvedValue([{ id: 1, content_embedding: new Float32Array(3) }]),
    fetchFactsByIds: vi.fn().mockResolvedValue([{ id: 1, content: 'fact', date_created: null }]),
    writeBatch: vi.fn().mockResolvedValue({ written_ops: 2 }),
    getConversationHistory: vi.fn().mockResolvedValue([]),
  };
}

/** Triggers lazy engine creation and returns the mock stub instance. */
async function bootEngine(engine: NativeEngine) {
  await engine.retrieve(RETRIEVE_REQ);
  return (MemoriEngine as any).mock.results.at(-1).value;
}

describe('NativeEngine', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // Construction
  // -------------------------------------------------------------------------

  it('hasStorage is false when no storageBridge is provided', () => {
    expect(new NativeEngine().hasStorage).toBe(false);
  });

  it('hasStorage is true when a storageBridge is provided', () => {
    expect(new NativeEngine(makeBridge()).hasStorage).toBe(true);
  });

  it('lazily constructs MemoriEngine on first use', async () => {
    const engine = new NativeEngine();
    await engine.retrieve(RETRIEVE_REQ);
    expect(MemoriEngine).toHaveBeenCalledTimes(1);
  });

  it('reuses the same MemoriEngine instance on subsequent calls', async () => {
    const engine = new NativeEngine();
    await engine.retrieve(RETRIEVE_REQ);
    await engine.retrieve(RETRIEVE_REQ);
    expect(MemoriEngine).toHaveBeenCalledTimes(1);
  });

  // -------------------------------------------------------------------------
  // retrieve / recall
  // -------------------------------------------------------------------------

  it('retrieve() maps NAPI camelCase fields to snake_case', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.retrieve.mockResolvedValue([
      {
        id: 42,
        content: 'fact text',
        rankScore: 0.9,
        similarity: 0.8,
        dateCreated: '2024-01-01',
        summaries: [{ content: 'sum', dateCreated: '2024-01-01', entityFactId: 10, factId: 5 }],
      },
    ]);

    const rows = await engine.retrieve(RETRIEVE_REQ);
    expect(rows[0].id).toBe(42);
    expect(rows[0].rank_score).toBe(0.9);
    expect(rows[0].similarity).toBe(0.8);
    expect(rows[0].date_created).toBe('2024-01-01');
    expect(rows[0].summaries![0].date_created).toBe('2024-01-01');
    expect(rows[0].summaries![0].entity_fact_id).toBe(10);
  });

  it('recall() proxies to native engine recall', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.recall.mockResolvedValue('The user lives in Paris.');
    const result = await engine.recall(RETRIEVE_REQ);
    expect(result).toBe('The user lives in Paris.');
  });

  // -------------------------------------------------------------------------
  // embedTexts
  // -------------------------------------------------------------------------

  it('embedTexts() returns empty array for empty input without touching engine', () => {
    const engine = new NativeEngine();
    expect(engine.embedTexts([])).toEqual([]);
    expect(MemoriEngine).not.toHaveBeenCalled();
  });

  it('embedTexts() delegates to native engine', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.embedTexts.mockReturnValue([new Float32Array(3)]);
    const result = engine.embedTexts(['hello']);
    expect(result).toHaveLength(1);
  });

  it('embedTexts() returns [] and logs error if native throws', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.embedTexts.mockImplementation(() => {
      throw new Error('embed fail');
    });
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(engine.embedTexts(['hello'])).toEqual([]);
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  // -------------------------------------------------------------------------
  // submitAugmentation
  // -------------------------------------------------------------------------

  it('submitAugmentation() passes through to native engine', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.submitAugmentation.mockReturnValue('aug-id-123');

    const id = engine.submitAugmentation({
      entity_id: 'u-1',
      process_id: 'p-1',
      conversation_id: 'c-1',
      conversation_messages: [],
    });
    expect(id).toBe('aug-id-123');
    expect(instance.submitAugmentation).toHaveBeenCalledWith(
      expect.objectContaining({ entityId: 'u-1', processId: 'p-1' })
    );
  });

  // -------------------------------------------------------------------------
  // waitForAugmentation
  // -------------------------------------------------------------------------

  it('waitForAugmentation() returns false when engine was never started', async () => {
    const engine = new NativeEngine();
    expect(await engine.waitForAugmentation(100)).toBe(false);
  });

  it('waitForAugmentation() delegates to native engine once started', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.waitForAugmentation.mockResolvedValue(true);
    expect(await engine.waitForAugmentation(500)).toBe(true);
    expect(instance.waitForAugmentation).toHaveBeenCalledWith(500);
  });

  // -------------------------------------------------------------------------
  // shutdown
  // -------------------------------------------------------------------------

  it('shutdown() is a no-op when engine was never started', () => {
    const engine = new NativeEngine();
    expect(() => {
      engine.shutdown();
    }).not.toThrow();
  });

  it('shutdown() calls native shutdown and resets state', async () => {
    const engine = new NativeEngine(makeBridge());
    const instance = await bootEngine(engine);
    engine.shutdown();
    expect(instance.shutdown).toHaveBeenCalled();
    expect(engine.hasStorage).toBe(false);
  });

  // -------------------------------------------------------------------------
  // Bridge callbacks — no-storage path
  // -------------------------------------------------------------------------

  it('no-storage fetchEmbeddingsCb resolves with empty array', async () => {
    const engine = new NativeEngine();
    await bootEngine(engine);
    const instance = (MemoriEngine as any).mock.results.at(-1).value;
    // The fetchEmbeddingsCb is the 2nd arg passed to the MemoriEngine constructor
    const fetchCb: (err: Error | null, id: number, payload: string) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[1];
    fetchCb(null, 1, '{}');
    expect(instance.resolveEmbeddingsCallback).toHaveBeenCalledWith(1, []);
  });

  it('no-storage writeBatchCb resolves with zero written ops', async () => {
    const engine = new NativeEngine();
    await bootEngine(engine);
    const instance = (MemoriEngine as any).mock.results.at(-1).value;
    const writeCb: (err: Error | null, id: number, payload: string) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[3];
    writeCb(null, 2, JSON.stringify({ ops: [] }));
    expect(instance.resolveWriteCallback).toHaveBeenCalledWith(2, { writtenOps: 0 });
  });

  // -------------------------------------------------------------------------
  // Bridge callbacks — with-storage path
  // -------------------------------------------------------------------------

  it('storage writeBatchCb calls bridge.writeBatch and resolves with written ops', async () => {
    const bridge = makeBridge();
    const engine = new NativeEngine(bridge);
    await bootEngine(engine);
    const instance = (MemoriEngine as any).mock.results.at(-1).value;
    const writeCb: (err: Error | null, id: number, payload: string) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[3];

    writeCb(null, 7, JSON.stringify({ ops: [] }));
    await new Promise(process.nextTick);

    // bridge.writeBatch returns { written_ops: 2 }
    expect(instance.resolveWriteCallback).toHaveBeenCalledWith(7, { writtenOps: 2 });
  });

  it('storage fetchEmbeddingsCb resolves and maps result rows', async () => {
    const bridge = makeBridge();
    const engine = new NativeEngine(bridge);
    await bootEngine(engine);
    const instance = (MemoriEngine as any).mock.results.at(-1).value;
    const fetchCb: (err: Error | null, id: number, payload: string) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[1];

    fetchCb(null, 5, JSON.stringify({ entity_id: 'u-1', limit: 5 }));
    await new Promise(process.nextTick);

    expect(instance.resolveEmbeddingsCallback).toHaveBeenCalledWith(
      5,
      expect.arrayContaining([expect.objectContaining({ id: 1 })])
    );
  });

  it('bridge error in fetchEmbeddings logs and calls fallback', async () => {
    const bridge = makeBridge();
    bridge.fetchEmbeddings.mockRejectedValue(new Error('fetch fail'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    const engine = new NativeEngine(bridge);
    await bootEngine(engine);
    const instance = (MemoriEngine as any).mock.results.at(-1).value;
    const fetchCb: (err: Error | null, id: number, payload: string) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[1];

    fetchCb(null, 5, JSON.stringify({ entity_id: 'u-1', limit: 5 }));
    await new Promise(process.nextTick);

    expect(consoleSpy).toHaveBeenCalled();
    expect(instance.resolveEmbeddingsCallback).toHaveBeenCalledWith(5, []);
    consoleSpy.mockRestore();
  });

  it('bridge error callback logs and returns early', async () => {
    const engine = new NativeEngine(makeBridge());
    await bootEngine(engine);
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const fetchCb: (err: Error | null, id: number, payload: string) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[1];
    fetchCb(new Error('bridge error'), 0, '{}');
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });
});
