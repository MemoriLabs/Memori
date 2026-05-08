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

function makeMysqlPool(overrides = {}) {
  return {
    execute: vi.fn().mockResolvedValue([[{ id: 1 }], []]),
    query: vi.fn().mockResolvedValue({}),
    release: vi.fn(),
    ...overrides,
  };
}

describe('MysqlAdapter', () => {
  it('execute() returns the first element of the result tuple', async () => {
    const pool = makeMysqlPool({ execute: vi.fn().mockResolvedValue([[{ id: 42 }], []]) });
    const adapter = new MysqlAdapter(pool);
    const rows = await adapter.execute('SELECT 1');
    expect(rows).toEqual([{ id: 42 }]);
  });

  it('execute() returns [] when rows is not an array', async () => {
    const pool = makeMysqlPool({ execute: vi.fn().mockResolvedValue([null, []]) });
    const adapter = new MysqlAdapter(pool);
    expect(await adapter.execute('INSERT')).toEqual([]);
  });

  it('begin() sends BEGIN query', async () => {
    const pool = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.begin();
    expect(pool.query).toHaveBeenCalledWith('BEGIN');
  });

  it('commit() sends COMMIT query', async () => {
    const pool = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.commit();
    expect(pool.query).toHaveBeenCalledWith('COMMIT');
  });

  it('rollback() sends ROLLBACK query', async () => {
    const pool = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    await adapter.rollback();
    expect(pool.query).toHaveBeenCalledWith('ROLLBACK');
  });

  it('close() calls release() if available (PoolConnection)', () => {
    const pool = makeMysqlPool();
    const adapter = new MysqlAdapter(pool);
    adapter.close();
    expect(pool.release).toHaveBeenCalled();
  });

  it('close() does nothing when release is not available (Pool)', () => {
    const pool = { execute: vi.fn(), query: vi.fn() };
    const adapter = new MysqlAdapter(pool);
    expect(() => {
      adapter.close();
    }).not.toThrow();
  });

  it('getDialect() returns "mysql"', () => {
    expect(new MysqlAdapter(makeMysqlPool()).getDialect()).toBe('mysql');
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
