import { describe, it, expect, beforeEach } from 'vitest';
import { Registry } from '../../src/storage/registry.js';
import { StorageAdapter, BaseDriver, SqlBindValue } from '../../src/storage/base.js';

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

class StubDriver extends BaseDriver {
  public requiresRollbackOnError = false;
  public migrations = {};
}

describe('Registry', () => {
  beforeEach(() => {
    // Register a fresh stub for each test so the registry has at least one entry.
    // Real adapters are already registered via side-effect imports in manager.ts,
    // but we test the registration mechanism independently here.
    Registry.registerAdapter(() => false, StubAdapter);
    Registry.registerDriver('stub', StubDriver);
  });

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

  describe('registerDriver / getDriver', () => {
    it('getDriver() returns correct driver for a known dialect', () => {
      const adapter = new StubAdapter();
      const driver = Registry.getDriver(adapter);
      expect(driver).toBeInstanceOf(StubDriver);
    });

    it('getDriver() throws for an unknown dialect', () => {
      class UnknownAdapter extends StubAdapter {
        getDialect() {
          return '__totally_unknown__';
        }
      }
      expect(() => Registry.getDriver(new UnknownAdapter())).toThrow(
        'Unsupported database dialect'
      );
    });
  });
});
