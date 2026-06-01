import { describe, it, expect, vi } from 'vitest';
import { SqliteAdapter } from '../../src/storage/adapters/sqlite.js';
import { MysqlAdapter } from '../../src/storage/adapters/mysql.js';
import { PostgresAdapter } from '../../src/storage/adapters/postgresql.js';
// ---------------------------------------------------------------------------
// SqliteAdapter
// ---------------------------------------------------------------------------

function makeSqliteDb(overrides: Record<string, unknown> = {}) {
  const stmt = { all: vi.fn().mockReturnValue([]), run: vi.fn(), reader: true };
  return {
    open: true,
    inTransaction: false,
    prepare: vi.fn().mockReturnValue(stmt),
    pragma: vi.fn(),
    close: vi.fn(),
    _stmt: stmt,
    ...overrides,
  };
}

describe('SqliteAdapter', () => {
  it('sets WAL mode and foreign keys on construction', () => {
    const db = makeSqliteDb();
    new SqliteAdapter(db);
    expect(db.pragma).toHaveBeenCalledWith('journal_mode = WAL');
    expect(db.pragma).toHaveBeenCalledWith('foreign_keys = ON');
  });

  it('execute() returns rows for reader statements', () => {
    const db = makeSqliteDb();
    db._stmt.reader = true;
    db._stmt.all.mockReturnValue([{ id: 1 }]);
    const adapter = new SqliteAdapter(db);
    const rows = adapter.execute('SELECT 1');
    expect(rows).toEqual([{ id: 1 }]);
  });

  it('execute() calls run() and returns [] for non-reader statements', () => {
    const db = makeSqliteDb();
    db._stmt.reader = false;
    const adapter = new SqliteAdapter(db);
    const rows = adapter.execute('INSERT INTO x VALUES (?)');
    expect(db._stmt.run).toHaveBeenCalled();
    expect(rows).toEqual([]);
  });

  it('execute() returns [] when db is closed', () => {
    const db = makeSqliteDb({ open: false });
    const adapter = new SqliteAdapter(db);
    expect(adapter.execute('SELECT 1')).toEqual([]);
  });

  it('begin() runs BEGIN when db is open and not in transaction', () => {
    const db = makeSqliteDb({ inTransaction: false });
    const adapter = new SqliteAdapter(db);
    adapter.begin();
    expect(db.prepare).toHaveBeenCalledWith('BEGIN');
    expect(db._stmt.run).toHaveBeenCalled();
  });

  it('begin() is a no-op when already in transaction', () => {
    const db = makeSqliteDb({ inTransaction: true });
    const adapter = new SqliteAdapter(db);
    adapter.begin();
    // pragma calls happen in constructor, but BEGIN prepare should not be called
    const beginCalls = db.prepare.mock.calls.filter((c: string[]) => c[0] === 'BEGIN');
    expect(beginCalls).toHaveLength(0);
  });

  it('commit() runs COMMIT when in transaction', () => {
    const db = makeSqliteDb({ inTransaction: true });
    const adapter = new SqliteAdapter(db);
    adapter.commit();
    expect(db.prepare).toHaveBeenCalledWith('COMMIT');
  });

  it('rollback() runs ROLLBACK when in transaction', () => {
    const db = makeSqliteDb({ inTransaction: true });
    const adapter = new SqliteAdapter(db);
    adapter.rollback();
    expect(db.prepare).toHaveBeenCalledWith('ROLLBACK');
  });

  it('close() does not close the user database — caller owns the lifecycle', () => {
    const db = makeSqliteDb();
    const adapter = new SqliteAdapter(db);
    adapter.close();
    expect(db.close).not.toHaveBeenCalled();
  });

  it('getDialect() returns "sqlite"', () => {
    const adapter = new SqliteAdapter(makeSqliteDb());
    expect(adapter.getDialect()).toBe('sqlite');
  });
});

// ---------------------------------------------------------------------------
// MysqlAdapter
// ---------------------------------------------------------------------------

function makeMysqlConn(overrides = {}) {
  return {
    execute: vi.fn().mockResolvedValue([[{ id: 1 }], []]),
    query: vi.fn().mockResolvedValue({}),
    beginTransaction: vi.fn().mockResolvedValue(undefined),
    commit: vi.fn().mockResolvedValue(undefined),
    rollback: vi.fn().mockResolvedValue(undefined),
    release: vi.fn(),
    ...overrides,
  };
}

function makeMysqlPool(connOverrides = {}, poolOverrides = {}) {
  const conn = makeMysqlConn(connOverrides);
  return {
    pool: {
      execute: vi.fn().mockResolvedValue([[{ id: 1 }], []]),
      query: vi.fn().mockResolvedValue({}),
      getConnection: vi.fn().mockResolvedValue(conn),
      ...poolOverrides,
    },
    conn,
  };
}

describe('MysqlAdapter', () => {
  it('execute() returns the first element of the result tuple', async () => {
    const { pool } = makeMysqlPool({}, { execute: vi.fn().mockResolvedValue([[{ id: 42 }], []]) });
    const adapter = new MysqlAdapter(pool);
    const rows = await adapter.execute('SELECT 1');
    expect(rows).toEqual([{ id: 42 }]);
  });

  it('execute() returns [] when rows is not an array', async () => {
    const { pool } = makeMysqlPool({}, { execute: vi.fn().mockResolvedValue([null, []]) });
    const adapter = new MysqlAdapter(pool);
    expect(await adapter.execute('INSERT')).toEqual([]);
  });

  it('begin() checks out a dedicated connection and calls beginTransaction()', async () => {
    const { pool, conn } = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.begin();
    expect(pool.getConnection).toHaveBeenCalled();
    expect(conn.beginTransaction).toHaveBeenCalled();
  });

  it('execute() routes through txConn after begin()', async () => {
    const { pool, conn } = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.begin();
    await adapter.execute('SELECT 1');
    expect(conn.execute).toHaveBeenCalledWith('SELECT 1', []);
    expect(pool.execute).not.toHaveBeenCalled();
  });

  it('commit() calls commit() on txConn and releases it', async () => {
    const { pool, conn } = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.begin();
    await adapter.commit();
    expect(conn.commit).toHaveBeenCalled();
    expect(conn.release).toHaveBeenCalled();
  });

  it('commit() releases connection and rethrows if commit() fails', async () => {
    const { pool, conn } = makeMysqlPool({
      commit: vi.fn().mockRejectedValue(new Error('deadlock')),
    });
    const adapter = new MysqlAdapter(pool);
    await adapter.begin();
    await expect(adapter.commit()).rejects.toThrow('deadlock');
    expect(conn.release).toHaveBeenCalled();
  });

  it('rollback() calls rollback() on txConn and releases it', async () => {
    const { pool, conn } = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.begin();
    await adapter.rollback();
    expect(conn.rollback).toHaveBeenCalled();
    expect(conn.release).toHaveBeenCalled();
  });

  it('rollback() is a no-op when no transaction is active', async () => {
    const { pool } = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await expect(adapter.rollback()).resolves.toBeUndefined();
  });

  it('close() releases txConn if a transaction is in flight', async () => {
    const { pool, conn } = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.begin();
    adapter.close();
    expect(conn.release).toHaveBeenCalled();
  });

  it('close() does nothing when no transaction is active', () => {
    const { pool } = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    expect(() => {
      adapter.close();
    }).not.toThrow();
  });

  it('getDialect() returns "mysql"', () => {
    const { pool } = makeMysqlPool();
    expect(new MysqlAdapter(pool).getDialect()).toBe('mysql');
  });

  it('accepts a direct connection (no getConnection) without throwing', () => {
    const conn = makeMysqlConn();
    expect(() => new MysqlAdapter(conn)).not.toThrow();
  });

  it('direct: execute() routes to the connection directly, not a pool', async () => {
    const conn = makeMysqlConn({ execute: vi.fn().mockResolvedValue([[{ id: 7 }], []]) });
    const adapter = new MysqlAdapter(conn);
    const rows = await adapter.execute('SELECT 1');
    expect(conn.execute).toHaveBeenCalledWith('SELECT 1', []);
    expect(rows).toEqual([{ id: 7 }]);
  });

  it('direct: begin() calls beginTransaction() on the connection, not getConnection()', async () => {
    const conn = makeMysqlConn();
    const adapter = new MysqlAdapter(conn);
    await adapter.begin();
    expect(conn.beginTransaction).toHaveBeenCalled();
    // No pool, so getConnection should never be called.
    expect((conn as Record<string, unknown>)['getConnection']).toBeUndefined();
  });

  it('direct: execute() still routes to the connection after begin()', async () => {
    const conn = makeMysqlConn();
    const adapter = new MysqlAdapter(conn);
    await adapter.begin();
    await adapter.execute('SELECT 1');
    expect(conn.execute).toHaveBeenCalledWith('SELECT 1', []);
  });

  it('direct: commit() calls commit() on the connection directly', async () => {
    const conn = makeMysqlConn();
    const adapter = new MysqlAdapter(conn);
    await adapter.begin();
    await adapter.commit();
    expect(conn.commit).toHaveBeenCalled();
    // release() is pool-only — must not be called on a direct connection.
    expect(conn.release).not.toHaveBeenCalled();
  });

  it('direct: rollback() calls rollback() on the connection directly', async () => {
    const conn = makeMysqlConn();
    const adapter = new MysqlAdapter(conn);
    await adapter.begin();
    await adapter.rollback();
    expect(conn.rollback).toHaveBeenCalled();
    expect(conn.release).not.toHaveBeenCalled();
  });

  it('direct: close() is a no-op — caller owns the connection lifecycle', () => {
    const conn = makeMysqlConn();
    const adapter = new MysqlAdapter(conn);
    expect(() => {
      adapter.close();
    }).not.toThrow();
    expect(conn.release).not.toHaveBeenCalled();
  });

  it('direct: getDialect() returns "mysql"', () => {
    expect(new MysqlAdapter(makeMysqlConn()).getDialect()).toBe('mysql');
  });
});

// ---------------------------------------------------------------------------
// PostgresAdapter
// ---------------------------------------------------------------------------

function makePoolClient() {
  return {
    query: vi.fn().mockResolvedValue({ rows: [] }),
    release: vi.fn(),
  };
}

function makePgPool(client = makePoolClient()) {
  return {
    query: vi.fn().mockResolvedValue({ rows: [{ id: 1 }] }),
    connect: vi.fn().mockResolvedValue(client),
    _client: client,
  };
}

describe('PostgresAdapter', () => {
  it('execute() uses pool.query() outside a transaction', async () => {
    const pool = makePgPool();
    const adapter = new PostgresAdapter(pool);
    const rows = await adapter.execute('SELECT 1');
    expect(pool.query).toHaveBeenCalledWith('SELECT 1', []);
    expect(rows).toEqual([{ id: 1 }]);
  });

  it('execute() uses txClient inside a transaction', async () => {
    const client = makePoolClient();
    client.query
      .mockResolvedValueOnce({ rows: [] }) // BEGIN
      .mockResolvedValueOnce({ rows: [{ id: 99 }] }); // SELECT
    const pool = makePgPool(client);
    const adapter = new PostgresAdapter(pool);
    await adapter.begin();
    const rows = await adapter.execute('SELECT 1');
    expect(client.query).toHaveBeenCalledWith('SELECT 1', []);
    expect(rows).toEqual([{ id: 99 }]);
  });

  it('begin() acquires a dedicated PoolClient and sends BEGIN', async () => {
    const pool = makePgPool();
    const adapter = new PostgresAdapter(pool);
    await adapter.begin();
    expect(pool.connect).toHaveBeenCalled();
    expect(pool._client.query).toHaveBeenCalledWith('BEGIN');
  });

  it('commit() sends COMMIT and releases the client', async () => {
    const client = makePoolClient();
    client.query.mockResolvedValue({ rows: [] });
    const pool = makePgPool(client);
    const adapter = new PostgresAdapter(pool);
    await adapter.begin();
    await adapter.commit();
    expect(client.query).toHaveBeenCalledWith('COMMIT');
    expect(client.release).toHaveBeenCalled();
  });

  it('rollback() sends ROLLBACK and destroys the client', async () => {
    const client = makePoolClient();
    client.query.mockResolvedValue({ rows: [] });
    const pool = makePgPool(client);
    const adapter = new PostgresAdapter(pool);
    await adapter.begin();
    await adapter.rollback();
    expect(client.query).toHaveBeenCalledWith('ROLLBACK');
    expect(client.release).toHaveBeenCalledWith(true);
  });

  it('rollback() releases client even if ROLLBACK query throws', async () => {
    const client = makePoolClient();
    client.query
      .mockResolvedValueOnce({ rows: [] }) // BEGIN
      .mockRejectedValueOnce(new Error('Connection terminated')); // ROLLBACK
    const pool = makePgPool(client);
    const adapter = new PostgresAdapter(pool);
    await adapter.begin();
    await expect(adapter.rollback()).resolves.toBeUndefined();
    expect(client.release).toHaveBeenCalledWith(true);
  });

  it('commit() destroys client and rethrows if COMMIT fails', async () => {
    const client = makePoolClient();
    client.query
      .mockResolvedValueOnce({ rows: [] }) // BEGIN
      .mockRejectedValueOnce(new Error('commit fail')); // COMMIT
    const pool = makePgPool(client);
    const adapter = new PostgresAdapter(pool);
    await adapter.begin();
    await expect(adapter.commit()).rejects.toThrow('commit fail');
    expect(client.release).toHaveBeenCalledWith(true);
  });

  it('close() releases txClient if a transaction is open', async () => {
    const client = makePoolClient();
    client.query.mockResolvedValue({ rows: [] });
    const pool = makePgPool(client);
    const adapter = new PostgresAdapter(pool);
    await adapter.begin();
    adapter.close();
    expect(client.release).toHaveBeenCalled();
  });

  it('close() is a no-op when no transaction is open', () => {
    const pool = makePgPool();
    const adapter = new PostgresAdapter(pool);
    expect(() => {
      adapter.close();
    }).not.toThrow();
    expect(pool._client.release).not.toHaveBeenCalled();
  });

  it('getDialect() returns "postgresql"', () => {
    expect(new PostgresAdapter(makePgPool()).getDialect()).toBe('postgresql');
  });
});
