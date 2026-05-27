/**
 * Integration test for the storageCall bridge using a real SQLite adapter.
 *
 * Does not load the native `.node` binary — exercises StorageManager +
 * SqliteAdapter the same way Rust does via acquire/execute/begin/commit/close.
 */
import { describe, it, expect, afterEach } from 'vitest';
import Database from 'better-sqlite3';
import { StorageManager } from '../../src/storage/manager.js';

function dispatchCall(manager: StorageManager, payload: object): Promise<object> {
  return new Promise((resolve) => {
    manager.handleStorageCall(0, JSON.stringify(payload), resolve);
  });
}

describe('storageCall protocol (SQLite)', () => {
  let db: Database.Database | undefined;
  let manager: StorageManager | undefined;

  afterEach(async () => {
    if (manager) await manager.close();
    if (db) db.close();
  });

  it('acquire → execute → close round-trips SQL through a real adapter', async () => {
    db = new Database(':memory:');
    manager = new StorageManager(() => db);

    const { conn_id } = (await dispatchCall(manager, { op: 'acquire' })) as {
      conn_id: number;
    };

    await dispatchCall(manager, {
      op: 'execute',
      conn_id,
      sql: 'CREATE TABLE bridge_test (id INTEGER PRIMARY KEY, name TEXT)',
      binds: [],
    });

    await dispatchCall(manager, {
      op: 'execute',
      conn_id,
      sql: 'INSERT INTO bridge_test (name) VALUES (?)',
      binds: [{ t: 'text', v: 'memori' }],
    });

    const select = (await dispatchCall(manager, {
      op: 'execute',
      conn_id,
      sql: 'SELECT id, name FROM bridge_test',
      binds: [],
    })) as { rows: Array<{ id: number; name: string }> };

    expect(select.rows).toEqual([{ id: 1, name: 'memori' }]);

    await dispatchCall(manager, { op: 'close', conn_id });
  });

  it('begin → execute → commit persists data in a transaction', async () => {
    db = new Database(':memory:');
    manager = new StorageManager(() => db);

    const { conn_id } = (await dispatchCall(manager, { op: 'acquire' })) as {
      conn_id: number;
    };

    await dispatchCall(manager, {
      op: 'execute',
      conn_id,
      sql: 'CREATE TABLE tx_test (value TEXT)',
      binds: [],
    });

    await dispatchCall(manager, { op: 'begin', conn_id });
    await dispatchCall(manager, {
      op: 'execute',
      conn_id,
      sql: 'INSERT INTO tx_test (value) VALUES (?)',
      binds: [{ t: 'text', v: 'committed' }],
    });
    await dispatchCall(manager, { op: 'commit', conn_id });

    const after = (await dispatchCall(manager, {
      op: 'execute',
      conn_id,
      sql: 'SELECT value FROM tx_test',
      binds: [],
    })) as { rows: Array<{ value: string }> };

    expect(after.rows).toEqual([{ value: 'committed' }]);
    await dispatchCall(manager, { op: 'close', conn_id });
  });
});
