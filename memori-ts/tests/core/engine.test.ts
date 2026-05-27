import { describe, it, expect, vi, beforeEach } from 'vitest';
import { NativeEngine } from '../../src/core/engine.js';
import { MemoriEngine } from '../../src/native/index.js';

// setup.ts mocks src/native/index.js — MemoriEngine is a vi.fn() that returns a stub object.

const RETRIEVE_REQ = { entity_id: 'u-1', query_text: 'q', dense_limit: 5, limit: 3 };

function makeStorageManager() {
  return {
    getDialect: vi.fn().mockReturnValue('sqlite'),
    handleStorageCall: vi.fn(),
    setEngineShutdown: vi.fn(),
    close: vi.fn(),
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

  it('hasStorage is false when no storageManager is provided', () => {
    expect(new NativeEngine().hasStorage).toBe(false);
  });

  it('hasStorage is true when a storageManager is provided', () => {
    expect(new NativeEngine(makeStorageManager() as any).hasStorage).toBe(true);
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

  it('passes dialect from storageManager to MemoriEngine constructor', async () => {
    const sm = makeStorageManager();
    sm.getDialect.mockReturnValue('postgresql');
    const engine = new NativeEngine(sm as any);
    await bootEngine(engine);
    // Third arg to MemoriEngine constructor is the dialect
    expect((MemoriEngine as any).mock.calls.at(-1)[2]).toBe('postgresql');
  });

  // -------------------------------------------------------------------------
  // build
  // -------------------------------------------------------------------------

  it('build() is a no-op when no storage is configured', async () => {
    const engine = new NativeEngine();
    await expect(engine.build()).resolves.toBeUndefined();
    expect(MemoriEngine).not.toHaveBeenCalled();
  });

  it('build() delegates to native engine when storage is configured', async () => {
    const engine = new NativeEngine(makeStorageManager() as any);
    const instance = await bootEngine(engine);
    instance.build.mockResolvedValue(undefined);
    await engine.build();
    expect(instance.build).toHaveBeenCalled();
  });

  // -------------------------------------------------------------------------
  // writeBatch
  // -------------------------------------------------------------------------

  it('writeBatch() returns zero when no storage is configured', async () => {
    const engine = new NativeEngine();
    const result = await engine.writeBatch({ ops: [] });
    expect(result.written_ops).toBe(0);
  });

  it('writeBatch() serializes batch to JSON and delegates to native engine', async () => {
    const engine = new NativeEngine(makeStorageManager() as any);
    const instance = await bootEngine(engine);
    instance.writeBatch.mockResolvedValue({ writtenOps: 3 });
    const result = await engine.writeBatch({ ops: [] });
    expect(result.written_ops).toBe(3);
    expect(instance.writeBatch).toHaveBeenCalledWith(JSON.stringify({ ops: [] }));
  });

  // -------------------------------------------------------------------------
  // getConversationHistory
  // -------------------------------------------------------------------------

  it('getConversationHistory() returns empty array when no storage is configured', async () => {
    const engine = new NativeEngine();
    const result = await engine.getConversationHistory('sess-1');
    expect(result).toEqual([]);
  });

  it('getConversationHistory() parses JSON response from native engine', async () => {
    const engine = new NativeEngine(makeStorageManager() as any);
    const instance = await bootEngine(engine);
    instance.getConversationHistory.mockResolvedValue(
      JSON.stringify([{ role: 'user', content: 'hello' }])
    );
    const result = await engine.getConversationHistory('sess-1');
    expect(result).toEqual([{ role: 'user', content: 'hello' }]);
  });

  // -------------------------------------------------------------------------
  // storageCallCb wiring
  // -------------------------------------------------------------------------

  it('storageCallCb routes calls to storageManager.handleStorageCall', async () => {
    const sm = makeStorageManager();
    const engine = new NativeEngine(sm as any);
    await bootEngine(engine);
    const instance = (MemoriEngine as any).mock.results.at(-1).value;

    // Extract the storageCallCb (2nd arg to MemoriEngine constructor)
    const storageCallCb: (err: Error | null, args: [number, string]) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[1];

    storageCallCb(null, [42, '{"op":"acquire"}']);
    expect(sm.handleStorageCall).toHaveBeenCalledWith(42, '{"op":"acquire"}', expect.any(Function));

    // The resolve callback should call resolveStorageCall on the engine
    const resolveFn = sm.handleStorageCall.mock.calls[0][2];
    resolveFn({ conn_id: 1 });
    expect(instance.resolveStorageCall).toHaveBeenCalledWith(42, JSON.stringify({ conn_id: 1 }));
  });

  it('storageCallCb logs and returns early on error', async () => {
    const sm = makeStorageManager();
    const engine = new NativeEngine(sm as any);
    await bootEngine(engine);
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    const storageCallCb: (err: Error | null, args: [number, string]) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[1];
    storageCallCb(new Error('bridge error'), [0, '']);
    expect(sm.handleStorageCall).not.toHaveBeenCalled();
    expect(consoleSpy).toHaveBeenCalled();
    consoleSpy.mockRestore();
  });

  it('no-storage storageCallCb resolves with NO_STORAGE error', async () => {
    const engine = new NativeEngine();
    await bootEngine(engine);
    const instance = (MemoriEngine as any).mock.results.at(-1).value;
    const noopCb: (err: Error | null, args: [number, string]) => void = (
      MemoriEngine as any
    ).mock.calls.at(-1)[1];
    noopCb(null, [9, '{"op":"acquire"}']);
    expect(instance.resolveStorageCall).toHaveBeenCalledWith(
      9,
      expect.stringContaining('NO_STORAGE')
    );
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

  it('embedTexts() returns empty array for empty input without touching engine', async () => {
    const engine = new NativeEngine();
    expect(await engine.embedTexts([])).toEqual([]);
    expect(MemoriEngine).not.toHaveBeenCalled();
  });

  it('embedTexts() delegates to native engine', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.embedTexts.mockResolvedValue([new Float32Array(3)]);
    const result = await engine.embedTexts(['hello']);
    expect(result).toHaveLength(1);
  });

  it('embedTexts() returns [] and logs error if native throws', async () => {
    const engine = new NativeEngine();
    const instance = await bootEngine(engine);
    instance.embedTexts.mockRejectedValue(new Error('embed fail'));
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    expect(await engine.embedTexts(['hello'])).toEqual([]);
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
    const sm = makeStorageManager();
    const engine = new NativeEngine(sm as any);
    const instance = await bootEngine(engine);
    engine.shutdown();
    expect(instance.shutdown).toHaveBeenCalled();
    expect(engine.hasStorage).toBe(false);
  });
});
