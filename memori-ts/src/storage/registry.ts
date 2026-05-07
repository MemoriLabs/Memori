import { StorageAdapter, BaseDriver, ConnFactory } from './base.js';

type MatcherFn = (conn: unknown) => boolean;
type AdapterConstructor = new (conn: unknown) => StorageAdapter;
type DriverConstructor = new (conn: StorageAdapter) => BaseDriver;

/**
 * Auto-discovery registry for storage adapters and dialect drivers.
 *
 * Adapters and drivers register themselves via side-effect imports in `StorageManager`.
 * `getAdapter` calls the factory once to obtain the connection, then inspects it to
 * find the right adapter class. The factory (not the connection itself) is the public
 * API boundary — Memori never holds a reference to pools or engines, only to the
 * individual connection the factory returned.
 */
export class Registry {
  private static adapters = new Map<MatcherFn, AdapterConstructor>();
  private static drivers = new Map<string, DriverConstructor>();

  /**
   * Registers a database adapter (e.g., pg, mysql2)
   */
  public static registerAdapter(matcher: MatcherFn, adapterClass: AdapterConstructor) {
    this.adapters.set(matcher, adapterClass);
  }

  /**
   * Registers a database driver syntax (e.g., postgresql, sqlite)
   */
  public static registerDriver(dialect: string, driverClass: DriverConstructor) {
    this.drivers.set(dialect, driverClass);
  }

  public static getAdapter(factory: ConnFactory): StorageAdapter {
    const conn = factory();

    for (const [matcher, AdapterClass] of this.adapters.entries()) {
      if (matcher(conn)) {
        return new AdapterClass(conn);
      }
    }
    throw new Error('Unsupported database connection object provided.');
  }

  public static getDriver(adapter: StorageAdapter): BaseDriver {
    const dialect = adapter.getDialect();
    const DriverClass = this.drivers.get(dialect);

    if (!DriverClass) {
      throw new Error(`Unsupported database dialect: ${dialect}`);
    }

    return new DriverClass(adapter);
  }
}
