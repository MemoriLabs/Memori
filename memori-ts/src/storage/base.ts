export type ConnFactory = () => unknown;

export type SqlBindValue =
  | string
  | number
  | boolean
  | null
  | Buffer
  | Uint8Array
  | (string | number)[];

export interface StorageAdapter {
  execute<T = Record<string, unknown>>(
    operation: string,
    binds?: SqlBindValue[]
  ): Promise<T[]> | T[];
  begin(): Promise<void> | void;
  commit(): Promise<void> | void;
  rollback(): Promise<void> | void;
  getDialect(): string;
  close(): Promise<void> | void;
  /**
   * Returns true when the adapter wraps a single shared handle that cannot safely
   * service concurrent connections (SQLite shared file, MySQL direct connection).
   * StorageManager serializes all acquire→close lifecycles for adapters that return true.
   */
  requiresSerialAccess?(): boolean;
}
