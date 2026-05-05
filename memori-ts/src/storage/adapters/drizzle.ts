import type { SQL } from 'drizzle-orm';
import type { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

export interface DrizzleInstance {
  execute?(query: SQL): Promise<unknown>;
  all?(query: SQL): unknown;
  run?(query: SQL): unknown;
}

function isDrizzleConnection(conn: unknown): conn is DrizzleInstance {
  return (
    typeof conn === 'object' &&
    conn !== null &&
    'select' in conn &&
    (('execute' in conn && typeof conn.execute === 'function') ||
      ('run' in conn && typeof conn.run === 'function') ||
      ('all' in conn && typeof conn.all === 'function'))
  );
}

export class DrizzleAdapter implements StorageAdapter {
  private readonly db: DrizzleInstance;

  constructor(conn: unknown) {
    this.db = conn as DrizzleInstance;
  }

  private async runQuery(query: SQL, rawSql: string): Promise<unknown> {
    if (typeof this.db.execute === 'function') {
      return await this.db.execute(query);
    }

    // SQLite requires .all() for operations returning data, and .run() for mutations
    const isSelect = /^\s*(SELECT|PRAGMA)\b/i.test(rawSql) || /\bRETURNING\b/i.test(rawSql);

    if (isSelect && typeof this.db.all === 'function') {
      return await this.db.all(query);
    } else if (typeof this.db.run === 'function') {
      return await this.db.run(query);
    }

    throw new Error('[Memori] Could not find a suitable execution method on Drizzle connection.');
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const placeholders = operation.match(/\$\d+|\?/g) ?? [];
    if (placeholders.length !== binds.length) {
      throw new Error(
        `[Memori] SQL placeholder count mismatch: expected ${placeholders.length}, got ${binds.length}`
      );
    }

    const { sql } = await import('drizzle-orm');
    const parts = operation.split(/\$\d+|\?/);
    let query = sql.raw(parts[0] || '');

    // Safely compose the Drizzle SQL object using standard template nesting
    for (let i = 0; i < binds.length; i++) {
      const val = binds[i];
      const nextPart = parts[i + 1] || '';
      query = sql`${query}${val}${sql.raw(nextPart)}`;
    }

    const result = await this.runQuery(query, operation);
    return this.normalizeResult<T>(result);
  }

  public async begin(): Promise<void> {
    const { sql } = await import('drizzle-orm');
    await this.runQuery(sql`BEGIN`, 'BEGIN');
  }

  public async commit(): Promise<void> {
    const { sql } = await import('drizzle-orm');
    await this.runQuery(sql`COMMIT`, 'COMMIT');
  }

  public async rollback(): Promise<void> {
    const { sql } = await import('drizzle-orm');
    await this.runQuery(sql`ROLLBACK`, 'ROLLBACK');
  }

  public getDialect(): string {
    const dbObj = this.db as unknown as Record<string, unknown>;

    // Safely check for nested dialect object without triggering unsafe-member-access
    if (
      typeof dbObj.dialect === 'object' &&
      dbObj.dialect !== null &&
      'tag' in dbObj.dialect &&
      typeof (dbObj.dialect as Record<string, unknown>).tag === 'string'
    ) {
      const tag = (dbObj.dialect as Record<string, unknown>).tag as string;
      if (tag === 'pg') return 'postgresql';
      if (tag === 'mysql') return 'mysql';
      if (tag === 'sqlite') return 'sqlite';
    }

    // Safely check constructor name
    const ctor = (this.db as { constructor?: { name?: string } }).constructor;
    const name = typeof ctor?.name === 'string' ? ctor.name : '';

    if (name.includes('Pg') || name.includes('Postgres')) return 'postgresql';
    if (name.includes('MySql')) return 'mysql';
    if (name.includes('SQLite') || name.includes('LibSQL')) return 'sqlite';

    throw new Error(`[Memori] Unable to determine dialect for Drizzle instance.`);
  }

  private normalizeResult<T>(res: unknown): T[] {
    if (!res) return [];

    if (typeof res === 'object' && 'rows' in res && Array.isArray(res.rows)) {
      return (res as { rows: unknown[] }).rows as T[];
    }

    if (Array.isArray(res) && res.length > 0 && Array.isArray(res[0])) {
      return res[0] as T[];
    }

    if (Array.isArray(res)) {
      return res as T[];
    }

    return [];
  }

  public close(): Promise<void> {
    return Promise.resolve();
  }
}

Registry.registerAdapter(isDrizzleConnection, DrizzleAdapter);
