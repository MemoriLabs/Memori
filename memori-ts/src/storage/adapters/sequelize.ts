import type { StorageAdapter, SqlBindValue } from '../base.js';
import { Registry } from '../registry.js';

export interface SequelizeTransaction {
  commit(): Promise<void>;
  rollback(): Promise<void>;
}

export interface SequelizeInstance {
  query(
    sql: string,
    options?: {
      replacements?: unknown[];
      bind?: unknown[];
      transaction?: SequelizeTransaction | null;
    }
  ): Promise<[unknown[], unknown]>;
  transaction(): Promise<SequelizeTransaction>;
  getDialect(): string;
}

function isSequelizeConnection(conn: unknown): conn is SequelizeInstance {
  return (
    typeof conn === 'object' &&
    conn !== null &&
    'getQueryInterface' in conn &&
    typeof conn.getQueryInterface === 'function' &&
    'transaction' in conn &&
    typeof (conn as { transaction: unknown }).transaction === 'function'
  );
}

export class SequelizeAdapter implements StorageAdapter {
  private tx: SequelizeTransaction | null = null;
  private readonly sequelize: SequelizeInstance;

  constructor(conn: unknown) {
    this.sequelize = conn as SequelizeInstance;
  }

  public async execute<T = Record<string, unknown>>(
    operation: string,
    binds: SqlBindValue[] = []
  ): Promise<T[]> {
    const options: {
      transaction: SequelizeTransaction | null;
      replacements?: unknown[];
      bind?: unknown[];
    } = {
      transaction: this.tx,
    };

    if (this.getDialect() === 'postgresql') {
      options.bind = binds;
    } else {
      options.replacements = binds;
    }

    const [results] = await this.sequelize.query(operation, options);
    return (Array.isArray(results) ? results : []) as T[];
  }

  public async begin(): Promise<void> {
    this.tx = await this.sequelize.transaction();
  }

  public async commit(): Promise<void> {
    if (this.tx) {
      await this.tx.commit();
      this.tx = null;
    }
  }

  public async rollback(): Promise<void> {
    if (this.tx) {
      await this.tx.rollback();
      this.tx = null;
    }
  }

  public getDialect(): string {
    const dialect = this.sequelize.getDialect();
    return dialect === 'postgres' ? 'postgresql' : dialect;
  }

  public close(): Promise<void> {
    return Promise.resolve();
  }
}

Registry.registerAdapter(isSequelizeConnection, SequelizeAdapter);
