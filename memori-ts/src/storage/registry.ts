import { StorageAdapter, BaseDriver } from './base.js';

type MatcherFn = (conn: unknown) => boolean;
type AdapterConstructor = new (conn: unknown) => StorageAdapter;
type DriverConstructor = new (conn: StorageAdapter) => BaseDriver;

/**
 * Auto-discovery registry for storage adapters and dialect drivers.
 *
 * Adapters and drivers register themselves via side-effect imports in `StorageManager`.
 * `getAdapter` inspects the raw connection at runtime to find the right adapter class.
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

  public static getAdapter(rawConn: unknown): StorageAdapter {
    // Some ORMs (e.g. Drizzle) export a factory function rather than a connection instance —
    // call it once to unwrap the actual connection before running matcher checks.
    const connToCheck = typeof rawConn === 'function' ? (rawConn as () => unknown)() : rawConn;

    for (const [matcher, AdapterClass] of this.adapters.entries()) {
      if (matcher(connToCheck)) {
        return new AdapterClass(connToCheck);
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
