import type { Database } from 'better-sqlite3';
import { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

function isSqliteConnection(conn: unknown): boolean {
  return (
    conn != null &&
    typeof (conn as Database).prepare === 'function' &&
    typeof (conn as Database).pragma === 'function'
  );
}

export class SqliteAdapter implements StorageAdapter {
  private client: Database;
  constructor(conn: unknown) {
    this.client = conn as Database;
    // WAL mode allows concurrent reads during a write — important for the async bridge callbacks
    this.client.pragma('journal_mode = WAL');
    // Enforce referential integrity at the DB level as a safety net
    this.client.pragma('foreign_keys = ON');
  }

  public execute<T = Record<string, unknown>>(operation: string, binds: SqlBindValue[] = []): T[] {
    if (!this.client.open) return [];
    const stmt = this.client.prepare(operation);
    // stmt.reader is true for SELECT-like statements that return rows; false for INSERT/UPDATE/DELETE
    return stmt.reader ? (stmt.all(...binds) as T[]) : (stmt.run(...binds), []);
  }

  public begin(): void {
    if (this.client.open && !this.client.inTransaction) this.client.prepare('BEGIN').run();
  }
  public commit(): void {
    if (this.client.open && this.client.inTransaction) this.client.prepare('COMMIT').run();
  }
  public rollback(): void {
    if (this.client.open && this.client.inTransaction) this.client.prepare('ROLLBACK').run();
  }
  public getDialect(): string {
    return 'sqlite';
  }
  public close(): void {
    if (this.client.open) this.client.close();
  }
}

Registry.registerAdapter(isSqliteConnection, SqliteAdapter);
