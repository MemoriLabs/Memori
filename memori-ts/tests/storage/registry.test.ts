import { describe, it, expect } from 'vitest';
import { Registry } from '../../src/storage/registry.js';
import { StorageAdapter, SqlBindValue } from '../../src/storage/base.js';

class StubAdapter implements StorageAdapter {
  execute(_op: string, _b?: SqlBindValue[]): [] {
    return [];
  }
  begin(): void {}
  commit(): void {}
  rollback(): void {}
  getDialect(): string {
    return 'stub';
  }
  close(): void {}
}

describe('Registry', () => {
  describe('registerAdapter / getAdapter', () => {
    it('getAdapter() calls the factory and matches the connection', () => {
      const obj = { isSpecial: true };
      Registry.registerAdapter((c) => (c as any).isSpecial === true, StubAdapter);
      const adapter = Registry.getAdapter(() => obj);
      expect(adapter).toBeInstanceOf(StubAdapter);
    });

    it('getAdapter() throws when no adapter matches', () => {
      expect(() => Registry.getAdapter(() => ({ __noMatchToken__: true }))).toThrow(
        'Unsupported database connection'
      );
    });
  });
});
