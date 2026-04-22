import { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

interface MysqlLike {
  execute(sql: string, binds?: SqlBindValue[]): Promise<[unknown[], unknown]>;
  query(sql: string): Promise<unknown>;
  end?(): Promise<void>;
  release?(): void;
}

function isMysqlConnection(conn: unknown): boolean {
  return (
    conn != null &&
    typeof (conn as MysqlLike).execute === 'function' &&
    typeof (conn as MysqlLike).query === 'function'
  );
}

export class MysqlAdapter implements StorageAdapter {
  private client: MysqlLike;

  constructor(conn: unknown) {
    this.client = conn as MysqlLike;
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const [rows] = await this.client.execute(operation, binds);
    return Array.isArray(rows) ? (rows as T[]) : [];
  }

  public async begin(): Promise<void> {
    await this.client.query('BEGIN');
  }
  public async commit(): Promise<void> {
    await this.client.query('COMMIT');
  }
  public async rollback(): Promise<void> {
    await this.client.query('ROLLBACK');
  }
  public getDialect(): string {
    return 'mysql';
  }

  public async close(): Promise<void> {
    if (typeof this.client.end === 'function') {
      await this.client.end();
    } else if (typeof this.client.release === 'function') {
      this.client.release();
    }
  }
}

Registry.registerAdapter(isMysqlConnection, MysqlAdapter);
