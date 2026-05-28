import { describe, it, expect, vi, beforeEach } from 'vitest';

// vi.mock is hoisted — all variables it references must be declared with vi.hoisted.
const { mockExecute, mockBegin, mockCommit, mockRollback, mockClose, mockGetDialect } = vi.hoisted(
  () => ({
    mockExecute: vi.fn().mockResolvedValue([{ id: 1 }]),
    mockBegin: vi.fn().mockResolvedValue(undefined),
    mockCommit: vi.fn().mockResolvedValue(undefined),
    mockRollback: vi.fn().mockResolvedValue(undefined),
    mockClose: vi.fn().mockResolvedValue(undefined),
    mockGetDialect: vi.fn().mockReturnValue('sqlite'),
  })
);

vi.mock('../../src/storage/registry.js', () => ({
  Registry: {
    registerAdapter: vi.fn(),
    getAdapter: vi.fn().mockReturnValue({
      execute: (...args: unknown[]) => mockExecute(...args),
      begin: (...args: unknown[]) => mockBegin(...args),
      commit: (...args: unknown[]) => mockCommit(...args),
      rollback: (...args: unknown[]) => mockRollback(...args),
      close: (...args: unknown[]) => mockClose(...args),
      getDialect: () => mockGetDialect(),
    }),
  },
}));

import { StorageManager } from '../../src/storage/manager.js';
import { Registry } from '../../src/storage/registry.js';

const makeFactory = () => () => ({});

/** Calls handleStorageCall and returns the result via the resolve callback. */
function dispatchCall(manager: StorageManager, payloadJson: string): Promise<object> {
  return new Promise((resolve) => {
    manager.handleStorageCall(0, payloadJson, resolve);
  });
}

describe('StorageManager', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  // -------------------------------------------------------------------------
  // Construction & dialect detection
  // -------------------------------------------------------------------------

  it('constructs without throwing', () => {
    expect(() => new StorageManager(makeFactory())).not.toThrow();
  });

  it('getDialect() returns the dialect from the adapter', () => {
    mockGetDialect.mockReturnValue('postgresql');
    const manager = new StorageManager(makeFactory());
    expect(manager.getDialect()).toBe('postgresql');
  });

  // -------------------------------------------------------------------------
  // handleStorageCall — acquire
  // -------------------------------------------------------------------------

  it('acquire op acquires a new adapter and returns a conn_id', async () => {
    const manager = new StorageManager(makeFactory());
    const result = await dispatchCall(manager, JSON.stringify({ op: 'acquire' }));
    expect(result).toHaveProperty('conn_id');
    expect((result as any).conn_id).toBeGreaterThan(0);
    expect(Registry.getAdapter).toHaveBeenCalledTimes(2); // once in constructor, once for acquire
  });

  it('consecutive acquire ops return distinct conn_ids', async () => {
    const manager = new StorageManager(makeFactory());
    const r1 = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const r2 = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    expect(r1.conn_id).not.toBe(r2.conn_id);
  });

  // -------------------------------------------------------------------------
  // handleStorageCall — execute
  // -------------------------------------------------------------------------

  it('execute op runs SQL on the connection and returns rows', async () => {
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT 1', binds: [] })
    )) as any;
    expect(result.rows).toEqual([{ id: 1 }]);
    expect(mockExecute).toHaveBeenCalledWith('SELECT 1', []);
  });

  it('execute op deserializes typed binds including Bytes as Buffer', async () => {
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const binds = [
      { t: 'int', v: 42 },
      { t: 'text', v: 'hello' },
      { t: 'null', v: null },
      { t: 'bytes', v: Buffer.from('data').toString('base64') },
    ];
    await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'INSERT INTO t VALUES (?,?,?,?)', binds })
    );
    const [, calledBinds] = mockExecute.mock.calls.at(-1)!;
    expect(calledBinds[0]).toBe(42);
    expect(calledBinds[1]).toBe('hello');
    expect(calledBinds[2]).toBeNull();
    expect(Buffer.isBuffer(calledBinds[3])).toBe(true);
  });

  it('execute op returns error object for unknown conn_id', async () => {
    const manager = new StorageManager(makeFactory());
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id: 9999, sql: 'SELECT 1', binds: [] })
    )) as any;
    expect(result.error).toBeDefined();
    expect(result.error.code).toBe('NO_CONN');
  });

  // -------------------------------------------------------------------------
  // handleStorageCall — begin / commit / rollback / close
  // -------------------------------------------------------------------------

  it('begin op calls adapter.begin() and returns { ok: true }', async () => {
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = await dispatchCall(manager, JSON.stringify({ op: 'begin', conn_id }));
    expect(result).toEqual({ ok: true });
    expect(mockBegin).toHaveBeenCalled();
  });

  it('commit op calls adapter.commit() and returns { ok: true }', async () => {
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = await dispatchCall(manager, JSON.stringify({ op: 'commit', conn_id }));
    expect(result).toEqual({ ok: true });
    expect(mockCommit).toHaveBeenCalled();
  });

  it('rollback op calls adapter.rollback() and returns { ok: true }', async () => {
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = await dispatchCall(manager, JSON.stringify({ op: 'rollback', conn_id }));
    expect(result).toEqual({ ok: true });
    expect(mockRollback).toHaveBeenCalled();
  });

  it('rollback op returns { ok: true } even for unknown conn_id (non-fatal)', async () => {
    const manager = new StorageManager(makeFactory());
    const result = await dispatchCall(manager, JSON.stringify({ op: 'rollback', conn_id: 9999 }));
    expect(result).toEqual({ ok: true });
  });

  it('close op calls adapter.close(), removes conn, and returns { ok: true }', async () => {
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = await dispatchCall(manager, JSON.stringify({ op: 'close', conn_id }));
    expect(result).toEqual({ ok: true });
    expect(mockClose).toHaveBeenCalled();

    // Subsequent execute on the closed conn_id should fail
    const execResult = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT 1', binds: [] })
    )) as any;
    expect(execResult.error).toBeDefined();
  });

  // -------------------------------------------------------------------------
  // handleStorageCall — error cases
  // -------------------------------------------------------------------------

  it('returns JSON_ERR for malformed JSON payload', async () => {
    const manager = new StorageManager(makeFactory());
    const result = (await dispatchCall(manager, 'not-valid-json')) as any;
    expect(result.error.code).toBe('JSON_ERR');
  });

  it('returns UNKNOWN_OP for unrecognised op', async () => {
    const manager = new StorageManager(makeFactory());
    const result = (await dispatchCall(manager, JSON.stringify({ op: 'explode' }))) as any;
    expect(result.error.code).toBe('UNKNOWN_OP');
  });

  it('returns ERR when execute adapter throws', async () => {
    mockExecute.mockRejectedValueOnce(new Error('db gone'));
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'BOOM', binds: [] })
    )) as any;
    expect(result.error.code).toBe('ERR');
    expect(result.error.message).toContain('db gone');
  });

  // -------------------------------------------------------------------------
  // P0 regression: Buffer rows must be base64-normalised before JSON transport
  // -------------------------------------------------------------------------

  it('execute op converts Buffer values in rows to base64 strings', async () => {
    const raw = Buffer.from([0x01, 0x02, 0x03, 0x04]);
    mockExecute.mockResolvedValueOnce([{ id: 1, content_embedding: raw }]);
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({
        op: 'execute',
        conn_id,
        sql: 'SELECT id, content_embedding FROM t',
        binds: [],
      })
    )) as any;
    expect(typeof result.rows[0].content_embedding).toBe('string');
    expect(result.rows[0].content_embedding).toBe(raw.toString('base64'));
  });

  it('execute op converts Uint8Array values in rows to base64 strings', async () => {
    const raw = new Uint8Array([0xde, 0xad, 0xbe, 0xef]);
    mockExecute.mockResolvedValueOnce([{ blob: raw }]);
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT blob FROM t', binds: [] })
    )) as any;
    expect(typeof result.rows[0].blob).toBe('string');
    expect(result.rows[0].blob).toBe(Buffer.from(raw).toString('base64'));
  });

  it('execute op converts BigInt row values to strings', async () => {
    mockExecute.mockResolvedValueOnce([{ id: BigInt('9007199254740993') }]);
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT id FROM t', binds: [] })
    )) as any;
    expect(typeof result.rows[0].id).toBe('string');
    expect(result.rows[0].id).toBe('9007199254740993');
  });

  it('execute op leaves non-binary row values untouched', async () => {
    mockExecute.mockResolvedValueOnce([{ id: 42, content: 'hello', score: 0.9 }]);
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT id, content, score FROM t', binds: [] })
    )) as any;
    expect(result.rows[0]).toEqual({ id: 42, content: 'hello', score: 0.9 });
  });

  // -------------------------------------------------------------------------
  // P2 regression: DB error codes must survive the TS bridge
  // -------------------------------------------------------------------------

  it('preserves the DB error code when the adapter throws with a code property', async () => {
    const err = Object.assign(new Error('serialization failure'), { code: '40001' });
    mockExecute.mockRejectedValueOnce(err);
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT 1', binds: [] })
    )) as any;
    expect(result.error.code).toBe('40001');
  });

  it('falls back to ERR code when the thrown error has no code property', async () => {
    mockExecute.mockRejectedValueOnce(new Error('generic failure'));
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;
    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT 1', binds: [] })
    )) as any;
    expect(result.error.code).toBe('ERR');
  });

  // -------------------------------------------------------------------------
  // P4 regression: lastUsed refresh keeps active connections alive past TTL
  // -------------------------------------------------------------------------

  it('refreshes lastUsed on execute so an active connection survives the TTL sweep', async () => {
    vi.useFakeTimers();
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;

    // Advance to just before the 30s TTL
    vi.advanceTimersByTime(29_000);

    // Touch the connection — refreshes lastUsed
    await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT 1', binds: [] })
    );

    // Advance 2s more: 31s since acquire, but only 2s since last use
    vi.advanceTimersByTime(2_000);

    // A new acquire triggers the sweep — our connection should survive
    await dispatchCall(manager, JSON.stringify({ op: 'acquire' }));

    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT 1', binds: [] })
    )) as any;
    expect(result.error).toBeUndefined();

    vi.useRealTimers();
  });

  it('sweeps a connection that has been idle past the TTL', async () => {
    vi.useFakeTimers();
    const manager = new StorageManager(makeFactory());
    const { conn_id } = (await dispatchCall(manager, JSON.stringify({ op: 'acquire' }))) as any;

    // Advance past TTL with no activity
    vi.advanceTimersByTime(31_000);

    // New acquire triggers sweep — idle connection should be removed
    await dispatchCall(manager, JSON.stringify({ op: 'acquire' }));

    const result = (await dispatchCall(
      manager,
      JSON.stringify({ op: 'execute', conn_id, sql: 'SELECT 1', binds: [] })
    )) as any;
    expect(result.error).toBeDefined();
    expect(result.error.code).toBe('NO_CONN');

    vi.useRealTimers();
  });
});
