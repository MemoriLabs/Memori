/**
 * Regression test: concurrent acquires on a MySQL direct connection must be serialized.
 *
 * When a user passes conn: myDirectConn, every acquire returns the same underlying
 * connection object. Without serialization, two concurrent write batches could interleave
 * beginTransaction/commit/rollback on that one handle, causing data corruption.
 * StorageManager prevents this with the same serialQueue used for SQLite.
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { StorageManager } from '../../src/storage/manager.js';

function makeMysqlDirectConn() {
  return {
    execute: vi.fn().mockResolvedValue([[], []]),
    query: vi.fn().mockResolvedValue({}),
    beginTransaction: vi.fn().mockResolvedValue(undefined),
    commit: vi.fn().mockResolvedValue(undefined),
    rollback: vi.fn().mockResolvedValue(undefined),
    release: vi.fn(),
  };
}

function dispatchCall(manager: StorageManager, payload: object): Promise<object> {
  return new Promise((resolve) => {
    manager.handleStorageCall(0, JSON.stringify(payload), resolve);
  });
}

describe('storageCall serialization (MySQL direct connection)', () => {
  let manager: StorageManager | undefined;

  afterEach(async () => {
    if (manager) await manager.close();
  });

  it('second acquire resolves only after the first connection is closed', async () => {
    const conn = makeMysqlDirectConn();
    manager = new StorageManager(() => conn);

    // Fire both acquires concurrently.
    const acq1Promise = dispatchCall(manager, { op: 'acquire' });
    const acq2Promise = dispatchCall(manager, { op: 'acquire' });

    // First acquire resolves immediately (it was first in the queue).
    const { conn_id: id1 } = (await acq1Promise) as { conn_id: number };

    // Close the first connection — this releases the serial lock.
    await dispatchCall(manager, { op: 'close', conn_id: id1 });

    // Now the second acquire can resolve.
    const { conn_id: id2 } = (await acq2Promise) as { conn_id: number };
    await dispatchCall(manager, { op: 'close', conn_id: id2 });

    expect(id1).not.toBe(id2);
  });

  it('first transaction fully completes before the second begins', async () => {
    const order: string[] = [];
    const conn = makeMysqlDirectConn();
    conn.beginTransaction.mockImplementation(async () => {
      order.push('beginTransaction');
    });
    conn.commit.mockImplementation(async () => {
      order.push('commit');
    });

    manager = new StorageManager(() => conn);

    const acq1 = dispatchCall(manager, { op: 'acquire' });
    const acq2 = dispatchCall(manager, { op: 'acquire' });

    const { conn_id: id1 } = (await acq1) as { conn_id: number };
    await dispatchCall(manager, { op: 'begin', conn_id: id1 });
    await dispatchCall(manager, { op: 'commit', conn_id: id1 });
    await dispatchCall(manager, { op: 'close', conn_id: id1 });
    order.push('close-1');

    const { conn_id: id2 } = (await acq2) as { conn_id: number };
    await dispatchCall(manager, { op: 'begin', conn_id: id2 });
    order.push('begin-2');
    await dispatchCall(manager, { op: 'close', conn_id: id2 });

    // The second begin must come after the first close.
    expect(order.indexOf('close-1')).toBeLessThan(order.indexOf('begin-2'));
    // Sanity-check interleaving didn't happen: first tx committed before second begin.
    expect(order).toEqual(['beginTransaction', 'commit', 'close-1', 'beginTransaction', 'begin-2']);
  });
});
