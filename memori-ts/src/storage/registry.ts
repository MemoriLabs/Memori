import { StorageAdapter, ConnFactory } from './base.js';

type MatcherFn = (conn: unknown) => boolean;
type AdapterConstructor = new (conn: unknown) => StorageAdapter;

/**
 * Auto-discovery registry for storage adapters.
 *
 * Adapters register themselves via side-effect imports in `StorageManager`.
 * `getAdapter` calls the factory once to obtain the connection, then inspects it
 * to find the right adapter class. The factory (not the connection itself) is the
 * public API boundary — Memori never holds a reference to pools or engines, only
 * to the individual connection the factory returned.
 */
export class Registry {
  private static adapters = new Map<MatcherFn, AdapterConstructor>();

  public static registerAdapter(matcher: MatcherFn, adapterClass: AdapterConstructor) {
    this.adapters.set(matcher, adapterClass);
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
}
