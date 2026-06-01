import { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

interface MysqlPool {
  execute(sql: string, binds?: SqlBindValue[]): Promise<[unknown[], unknown]>;
  query(sql: string): Promise<unknown>;
  getConnection(): Promise<MysqlConnection>;
  end?(): Promise<void>;
}

interface MysqlConnection {
  execute(sql: string, binds?: SqlBindValue[]): Promise<[unknown[], unknown]>;
  query(sql: string): Promise<unknown>;
  beginTransaction(): Promise<void>;
  commit(): Promise<void>;
  rollback(): Promise<void>;
  release(): void;
  destroy?(): void;
}

function isMysqlPool(conn: unknown): conn is MysqlPool {
  if (conn == null) return false;
  const c = conn as Record<string, unknown>;
  return (
    typeof c['execute'] === 'function' &&
    typeof c['query'] === 'function' &&
    typeof c['getConnection'] === 'function'
  );
}

function isMysqlDirectConnection(conn: unknown): conn is MysqlConnection {
  if (conn == null) return false;
  const c = conn as Record<string, unknown>;
  return (
    typeof c['execute'] === 'function' &&
    typeof c['beginTransaction'] === 'function' &&
    typeof c['commit'] === 'function' &&
    typeof c['rollback'] === 'function'
  );
}

function isMysqlCompatible(conn: unknown): boolean {
  return isMysqlPool(conn) || isMysqlDirectConnection(conn);
}

export class MysqlAdapter implements StorageAdapter {
  // Exactly one of these is set based on what the user passed.
  private readonly pool: MysqlPool | null;
  private readonly directConn: MysqlConnection | null;
  // Pool mode only: the checked-out connection held for the transaction duration.
  private txConn: MysqlConnection | null = null;

  private getPool(): MysqlPool {
    if (!this.pool) throw new Error('[Memori] MysqlAdapter: pool is not set');
    return this.pool;
  }

  constructor(conn: unknown) {
    if (isMysqlPool(conn)) {
      this.pool = conn;
      this.directConn = null;
    } else {
      this.pool = null;
      this.directConn = conn as MysqlConnection;
    }
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    let rows: unknown;
    // Direct connection: always execute on it
    // the adapter calls conn.cursor().execute() on the same connection throughout.
    // Pool: use the checked-out txConn during a transaction, pool otherwise.
    if (this.directConn) {
      [rows] = await this.directConn.execute(operation, binds);
    } else if (this.txConn) {
      [rows] = await this.txConn.execute(operation, binds);
    } else {
      [rows] = await this.getPool().execute(operation, binds);
    }
    return Array.isArray(rows) ? (rows as T[]) : [];
  }

  public async begin(): Promise<void> {
    if (this.directConn) {
      // Direct connection: call beginTransaction() on the connection itself
      await this.directConn.beginTransaction();
    } else {
      // Pool: check out a dedicated connection for the transaction lifetime.
      const conn = await this.getPool().getConnection();
      try {
        await conn.beginTransaction();
      } catch (e) {
        conn.release();
        throw e;
      }
      this.txConn = conn;
    }
  }

  public async commit(): Promise<void> {
    if (this.directConn) {
      await this.directConn.commit();
    } else if (this.txConn) {
      const conn = this.txConn;
      try {
        await conn.commit();
        this.txConn = null;
        conn.release();
      } catch (e) {
        // Commit failed — transaction state is unknown; don't return connection as clean.
        this.txConn = null;
        try {
          await conn.rollback();
        } catch {
          // ignore secondary failure
        }
        if (conn.destroy) conn.destroy();
        else conn.release();
        throw e;
      }
    }
  }

  public async rollback(): Promise<void> {
    if (this.directConn) {
      try {
        await this.directConn.rollback();
      } catch {
        // non-fatal
      }
    } else if (this.txConn) {
      const conn = this.txConn;
      this.txConn = null;
      let failed = false;
      try {
        await conn.rollback();
      } catch {
        failed = true;
      } finally {
        if (failed && conn.destroy) conn.destroy();
        else conn.release();
      }
    }
  }

  public getDialect(): string {
    return 'mysql';
  }

  public requiresSerialAccess(): boolean {
    // A direct connection is a single shared handle — concurrent acquires would race.
    // A pool safely issues getConnection() for each acquire, so no serialization needed.
    return this.directConn !== null;
  }

  public close(): void {
    // Direct connection: nothing to release — caller manages its lifetime.
    // Pool: release any orphaned checked-out connection - never call pool.end().
    if (this.txConn) {
      this.txConn.release();
      this.txConn = null;
    }
  }
}

Registry.registerAdapter(isMysqlCompatible, MysqlAdapter);
