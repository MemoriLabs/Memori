import type { PoolClient, Pool } from 'pg';
import { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

function isPostgresConnection(conn: unknown): boolean {
  return (
    conn != null &&
    typeof (conn as Pool).query === 'function' &&
    typeof (conn as Pool).connect === 'function' &&
    typeof (conn as { execute?: unknown }).execute !== 'function'
  );
}

export class PostgresAdapter implements StorageAdapter {
  private readonly pool: Pool;
  private txConn: PoolClient | null = null;

  constructor(conn: unknown) {
    this.pool = conn as Pool;
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const client = this.txConn ?? this.pool;
    const result = await client.query(operation, binds);
    return result.rows as T[];
  }

  public async begin(): Promise<void> {
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
    } catch (e) {
      // BEGIN failed — destroy rather than return to pool since state is unknown.
      client.release(true);
      throw e;
    }
    this.txConn = client;
  }

  public async commit(): Promise<void> {
    if (this.txConn) {
      const client = this.txConn;
      this.txConn = null;
      try {
        await client.query('COMMIT');
        client.release();
      } catch (e) {
        // Connection may be in an unknown state after a failed COMMIT — destroy it.
        client.release(true);
        throw e;
      }
    }
  }

  public async rollback(): Promise<void> {
    if (this.txConn) {
      const client = this.txConn;
      this.txConn = null;
      try {
        await client.query('ROLLBACK');
      } catch {
        // Connection may be terminated — that's fine, we're rolling back anyway.
      } finally {
        client.release(true);
      }
    }
  }

  public getDialect(): string {
    return 'postgresql';
  }

  public close(): void {
    // Release any open transaction client — never call pool.end(), caller owns pool lifecycle.
    if (this.txConn) {
      this.txConn.release();
      this.txConn = null;
    }
  }
}

Registry.registerAdapter(isPostgresConnection, PostgresAdapter);
