import { describe, it, expect, vi, beforeEach } from 'vitest';
import { Builder } from '../../src/storage/builder.js';
import { Config } from '../../src/core/config.js';
import type { StorageAdapter, BaseDriver, Migration } from '../../src/storage/base.js';

function makeAdapter(): StorageAdapter {
  return {
    execute: vi.fn().mockResolvedValue([]),
    begin: vi.fn().mockResolvedValue(undefined),
    commit: vi.fn().mockResolvedValue(undefined),
    rollback: vi.fn().mockResolvedValue(undefined),
    getDialect: vi.fn().mockReturnValue('sqlite'),
    close: vi.fn(),
  };
}

function makeDriver(
  currentVersion: number | null,
  migrations: Partial<Record<number, Migration[]>> = {},
  requiresRollback = false
): BaseDriver {
  return {
    requiresRollbackOnError: requiresRollback,
    migrations,
    schema: {
      version: {
        read: vi.fn().mockResolvedValue(currentVersion),
        delete: vi.fn().mockResolvedValue(undefined),
        create: vi.fn().mockResolvedValue(undefined),
      },
    },
  } as unknown as BaseDriver;
}

describe('Builder', () => {
  let config: Config;

  beforeEach(() => {
    config = new Config();
  });

  it('does nothing when schema is already at max version', async () => {
    const adapter = makeAdapter();
    const driver = makeDriver(2, {
      1: [{ description: 'v1', operation: 'CREATE TABLE a (id INT)' }],
      2: [{ description: 'v2', operation: 'CREATE TABLE b (id INT)' }],
    });
    const builder = new Builder(config, adapter, driver).disableBanner();
    await builder.execute();
    expect(adapter.begin).not.toHaveBeenCalled();
  });

  it('runs pending migrations and updates the version', async () => {
    const adapter = makeAdapter();
    const driver = makeDriver(0, {
      1: [{ description: 'v1', operation: 'CREATE TABLE a (id INT)' }],
    });
    const builder = new Builder(config, adapter, driver).disableBanner();
    await builder.execute();

    expect(adapter.begin).toHaveBeenCalledTimes(2); // 1 migration + version update
    expect(adapter.execute).toHaveBeenCalledWith('CREATE TABLE a (id INT)');
    expect(adapter.commit).toHaveBeenCalledTimes(2);
    expect(driver.schema.version.create as any).toHaveBeenCalledWith(1);
  });

  it('handles a migration with multiple operations', async () => {
    const adapter = makeAdapter();
    const driver = makeDriver(0, {
      1: [
        { description: 'v1', operations: ['CREATE TABLE a (id INT)', 'CREATE INDEX i ON a(id)'] },
      ],
    });
    const builder = new Builder(config, adapter, driver).disableBanner();
    await builder.execute();

    expect(adapter.execute).toHaveBeenCalledWith('CREATE TABLE a (id INT)');
    expect(adapter.execute).toHaveBeenCalledWith('CREATE INDEX i ON a(id)');
  });

  it('rolls back and resets to version 0 when schema read throws and requiresRollbackOnError=true', async () => {
    const adapter = makeAdapter();
    const driver = makeDriver(
      null,
      {
        1: [{ description: 'v1', operation: 'CREATE TABLE a (id INT)' }],
      },
      true
    );
    // Make schema version read throw — simulates table not existing yet
    (driver.schema.version.read as any).mockRejectedValue(new Error('no such table'));

    const builder = new Builder(config, adapter, driver).disableBanner();
    await builder.execute();

    expect(adapter.rollback).toHaveBeenCalled();
    // Should still run the migration from version 0
    expect(adapter.execute).toHaveBeenCalledWith('CREATE TABLE a (id INT)');
  });

  it('does not roll back when requiresRollbackOnError=false and schema read throws', async () => {
    const adapter = makeAdapter();
    const driver = makeDriver(null, {}, false);
    (driver.schema.version.read as any).mockRejectedValue(new Error('no table'));

    const builder = new Builder(config, adapter, driver).disableBanner();
    await builder.execute();

    expect(adapter.rollback).not.toHaveBeenCalled();
  });

  it('applies multiple migration batches in order', async () => {
    const adapter = makeAdapter();
    const driver = makeDriver(1, {
      1: [{ description: 'v1', operation: 'CREATE TABLE a (id INT)' }],
      2: [{ description: 'v2', operation: 'CREATE TABLE b (id INT)' }],
      3: [{ description: 'v3', operation: 'CREATE TABLE c (id INT)' }],
    });
    const builder = new Builder(config, adapter, driver).disableBanner();
    await builder.execute();

    const executedSql = (adapter.execute as any).mock.calls.map((c: string[]) => c[0]);
    expect(executedSql).not.toContain('CREATE TABLE a (id INT)');
    expect(executedSql).toContain('CREATE TABLE b (id INT)');
    expect(executedSql).toContain('CREATE TABLE c (id INT)');
    expect(driver.schema.version.create as any).toHaveBeenCalledWith(3);
  });
});
