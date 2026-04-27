import type { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

export interface TypeOrmQueryRunner {
  isTransactionActive: boolean;
  query(query: string, parameters?: unknown[]): Promise<unknown>;
  startTransaction(): Promise<void>;
  commitTransaction(): Promise<void>;
  rollbackTransaction(): Promise<void>;
  release(): Promise<void>;
}

export interface TypeOrmDataSource {
  options: { type: string };
  createQueryRunner(): TypeOrmQueryRunner;
}

function isTypeOrmConnection(conn: unknown): conn is TypeOrmDataSource {
  return (
    typeof conn === 'object' &&
    conn !== null &&
    'createQueryRunner' in conn &&
    typeof conn.createQueryRunner === 'function' &&
    'options' in conn &&
    typeof (conn as { options: unknown }).options === 'object'
  );
}

export class TypeOrmAdapter implements StorageAdapter {
  private readonly queryRunner: TypeOrmQueryRunner;
  private readonly dataSource: TypeOrmDataSource;

  constructor(conn: unknown) {
    this.dataSource = conn as TypeOrmDataSource;
    this.queryRunner = this.dataSource.createQueryRunner();
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const result = await this.queryRunner.query(operation, binds);
    return (Array.isArray(result) ? result : []) as T[];
  }

  public async begin(): Promise<void> {
    if (!this.queryRunner.isTransactionActive) {
      await this.queryRunner.startTransaction();
    }
  }

  public async commit(): Promise<void> {
    // FIX: Check if active before committing
    if (this.queryRunner.isTransactionActive) {
      await this.queryRunner.commitTransaction();
    }
  }

  public async rollback(): Promise<void> {
    // FIX: Check if active before rolling back
    if (this.queryRunner.isTransactionActive) {
      await this.queryRunner.rollbackTransaction();
    }
  }

  public getDialect(): string {
    const type = this.dataSource.options.type;
    if (type === 'postgres' || type === 'cockroachdb') return 'postgresql';
    if (type === 'mysql' || type === 'mariadb') return 'mysql';
    if (type === 'sqlite' || type === 'better-sqlite3') return 'sqlite';
    return type;
  }

  public async close(): Promise<void> {
    await this.queryRunner.release();
  }
}

Registry.registerAdapter(isTypeOrmConnection, TypeOrmAdapter);
