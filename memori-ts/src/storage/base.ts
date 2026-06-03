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
  // Return true for single-handle adapters (SQLite, MySQL direct) — StorageManager will serialize acquires.
  requiresSerialAccess?(): boolean;
}
