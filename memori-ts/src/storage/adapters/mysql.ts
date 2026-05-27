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

function isMysqlConnection(conn: unknown): boolean {
  return (
    conn != null &&
    typeof (conn as MysqlPool).execute === 'function' &&
    typeof (conn as MysqlPool).query === 'function' &&
    typeof (conn as MysqlPool).getConnection === 'function'
  );
}

export class MysqlAdapter implements StorageAdapter {
  private readonly pool: MysqlPool;
  private txConn: MysqlConnection | null = null;

  constructor(conn: unknown) {
    this.pool = conn as MysqlPool;
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const client = this.txConn ?? this.pool;
    const [rows] = await client.execute(operation, binds);
    return Array.isArray(rows) ? (rows as T[]) : [];
  }

  public async begin(): Promise<void> {
    this.txConn = await this.pool.getConnection();
    await this.txConn.beginTransaction();
  }

  public async commit(): Promise<void> {
    if (this.txConn) {
      const conn = this.txConn;
      this.txConn = null;
      try {
        await conn.commit();
      } finally {
        conn.release();
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
        conn.release();
      }
    }
  }

  public getDialect(): string {
    return 'mysql';
  }

  public close(): void {
    // Release any checked-out transaction connection — never call pool.end(), caller owns pool lifecycle.
    if (this.txConn) {
      this.txConn.release();
      this.txConn = null;
    }
  }
}

Registry.registerAdapter(isMysqlConnection, MysqlAdapter);
