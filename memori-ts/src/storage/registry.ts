import { StorageAdapter, ConnFactory } from './base.js';

type MatcherFn = (conn: unknown) => boolean;
type AdapterConstructor = new (conn: unknown) => StorageAdapter;

// Adapters self-register via side-effect imports; getAdapter probes the connection to pick the right one.
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
