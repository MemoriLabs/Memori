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
}

// Accepts both a mysql2/promise Pool (has getConnection) and a direct
// PoolConnection / Connection (has beginTransaction but no getConnection).
function isMysqlConnection(conn: unknown): boolean {
  if (conn == null) return false;
  const c = conn as MysqlPool & MysqlConnection;
  return (
    typeof c.execute === 'function' &&
    typeof c.query === 'function' &&
    (typeof c.getConnection === 'function' || typeof c.beginTransaction === 'function')
  );
}

export class MysqlAdapter implements StorageAdapter {
  private readonly conn: MysqlPool | MysqlConnection;
  // True when the factory returned a Pool (has getConnection); false for direct connections.
  private readonly isPool: boolean;
  private txConn: MysqlConnection | null = null;

  constructor(conn: unknown) {
    this.isPool = typeof (conn as MysqlPool).getConnection === 'function';
    this.conn = conn as MysqlPool | MysqlConnection;
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const client = this.txConn ?? (this.conn as MysqlConnection);
    const [rows] = await client.execute(operation, binds);
    return Array.isArray(rows) ? (rows as T[]) : [];
  }

  public async begin(): Promise<void> {
    if (this.isPool) {
      // Acquire into a local first; only store on the instance after the transaction
      // has started so a failed beginTransaction() doesn't leave an unreleased connection.
      const conn = await (this.conn as MysqlPool).getConnection();
      try {
        await conn.beginTransaction();
      } catch (e) {
        conn.release();
        throw e;
      }
      this.txConn = conn;
    } else {
      // Direct connection — begin transaction in-place; caller owns lifecycle.
      await (this.conn as MysqlConnection).beginTransaction();
      this.txConn = this.conn as MysqlConnection;
    }
  }

  public async commit(): Promise<void> {
    if (this.txConn) {
      const conn = this.txConn;
      this.txConn = null;
      try {
        await conn.commit();
      } finally {
        // Only release pool-checked-out connections; never release a direct connection.
        if (this.isPool) conn.release();
      }
    }
  }

  public async rollback(): Promise<void> {
    if (this.txConn) {
      const conn = this.txConn;
      this.txConn = null;
      try {
        await conn.rollback();
      } catch {
        // non-fatal
      } finally {
        if (this.isPool) conn.release();
      }
    }
  }

  public getDialect(): string {
    return 'mysql';
  }

  public close(): void {
    // Release any checked-out pool connection — never call pool.end() or release a direct connection.
    if (this.txConn) {
      if (this.isPool) this.txConn.release();
      this.txConn = null;
    }
  }
}

Registry.registerAdapter(isMysqlConnection, MysqlAdapter);
