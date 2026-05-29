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

function isMysqlConnection(conn: unknown): boolean {
  if (conn == null) return false;
  const c = conn as MysqlPool;
  return (
    typeof c.execute === 'function' &&
    typeof c.query === 'function' &&
    typeof c.getConnection === 'function'
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
    const [rows] = this.txConn
      ? await this.txConn.execute(operation, binds)
      : await this.pool.execute(operation, binds);
    return Array.isArray(rows) ? (rows as T[]) : [];
  }

  public async begin(): Promise<void> {
    const conn = await this.pool.getConnection();
    try {
      await conn.beginTransaction();
    } catch (e) {
      conn.release();
      throw e;
    }
    this.txConn = conn;
  }

  public async commit(): Promise<void> {
    if (this.txConn) {
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
    if (this.txConn) {
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

  public close(): void {
    // Release any checked-out pool connection — never call pool.end().
    if (this.txConn) {
      this.txConn.release();
      this.txConn = null;
    }
  }
}

Registry.registerAdapter(isMysqlConnection, MysqlAdapter);
