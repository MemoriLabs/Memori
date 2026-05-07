import type { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

export interface MikroOrmConnection {
  execute(query: string, params?: unknown[], method?: string, ctx?: unknown): Promise<unknown>;
  getPlatform(): { constructor: { name: string } };
}

export interface MikroOrmLike {
  getConnection(): MikroOrmConnection;
  begin(): Promise<void>;
  commit(): Promise<void>;
  rollback(): Promise<void>;
  execute?(query: string, params?: unknown[]): Promise<unknown>;
  getTransactionContext?(): unknown;
}

function isMikroOrmConnection(conn: unknown): conn is MikroOrmLike {
  return (
    typeof conn === 'object' &&
    conn !== null &&
    'getConnection' in conn &&
    typeof conn.getConnection === 'function' &&
    'getDriver' in conn &&
    typeof (conn as { getDriver: unknown }).getDriver === 'function'
  );
}

export class MikroOrmAdapter implements StorageAdapter {
  private readonly em: MikroOrmLike;
  private readonly connection: MikroOrmConnection;

  constructor(conn: unknown) {
    this.em = conn as MikroOrmLike;
    this.connection = this.em.getConnection();
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const sql = this.getDialect() === 'postgresql' ? operation.replace(/\$\d+/g, '?') : operation;

    let result: unknown;

    if (typeof this.em.execute === 'function') {
      result = await this.em.execute(sql, binds);
    } else {
      const ctx =
        typeof this.em.getTransactionContext === 'function'
          ? this.em.getTransactionContext()
          : undefined;
      result = await this.connection.execute(sql, binds, 'all', ctx);
    }

    return (Array.isArray(result) ? result : []) as T[];
  }

  public async begin(): Promise<void> {
    await this.em.begin();
  }

  public async commit(): Promise<void> {
    await this.em.commit();
  }

  public async rollback(): Promise<void> {
    await this.em.rollback();
  }

  public getDialect(): string {
    const platform = this.connection.getPlatform();
    const name = platform.constructor.name;

    if (name.includes('PostgreSql') || name.includes('Pg')) return 'postgresql';
    if (name.includes('MySql') || name.includes('MariaDb')) return 'mysql';
    if (name.includes('Sqlite')) return 'sqlite';

    throw new Error(
      `[Memori] Unable to determine dialect for MikroORM instance. Platform: ${name}`
    );
  }

  public close(): Promise<void> {
    return Promise.resolve();
  }
}

Registry.registerAdapter(isMikroOrmConnection, MikroOrmAdapter);
