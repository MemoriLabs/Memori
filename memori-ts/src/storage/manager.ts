import { StorageAdapter, ConnFactory, SqlBindValue } from './base.js';
import { Registry } from './registry.js';

// Side-effect imports: each module calls Registry.registerAdapter on load,
// so the Registry can auto-detect the connection type at runtime.
import './adapters/postgresql.js';
import './adapters/sqlite.js';
import './adapters/mysql.js';

type StorageCallPayload = {
  op: string;
  conn_id?: number;
  sql?: string;
  binds?: Array<{ t: string; v: unknown }>;
};

/**
 * Converts the `{ t, v }` tagged-union bind parameters that Rust sends over the
 * storageCall bridge into native JS values that adapters understand.
 * Binary embeddings arrive as base64-encoded strings and are decoded to Buffer.
 */
function deserializeBinds(binds: Array<{ t: string; v: unknown }>): SqlBindValue[] {
  return binds.map((bind) => {
    switch (bind.t) {
      case 'null':
        return null;
      case 'int':
        return bind.v as string;
      case 'float':
        return bind.v as number;
      case 'text':
        return bind.v as string;
      case 'bytes':
        return Buffer.from(bind.v as string, 'base64');
      default:
        console.warn(`[Memori] unknown bind type "${bind.t}" — treating as NULL`);
        return null;
    }
  });
}

/**
 * Thin bridge between the Rust storage layer and the user's database connection.
 *
 * Rust calls the `storageCall` TSFN with `(id, payloadJson)` for every storage
 * operation (acquire / execute / begin / commit / rollback / close). This class
 * dispatches those calls to the appropriate `StorageAdapter`, then resolves each
 * call back to Rust via `engine.resolveStorageCall(id, resultJson)`.
 *
 * Active connections are tracked in a `Map<connId, StorageAdapter>` keyed by the
 * numeric connection ID that Rust uses to route subsequent operations.
 */
/** Milliseconds of inactivity before an acquired-but-never-closed connection is considered orphaned. */
const CONN_TTL_MS = 30_000;
const SWEEP_INTERVAL_MS = 60_000;

/**
 * Normalizes a single DB row value for JSON transport back to Rust.
 * Buffer/Uint8Array values (e.g. BLOB/BYTEA columns) are converted to base64 strings
 * so that Rust's `row["field"].as_str()` can read them correctly.
 */
function normalizeRowValue(v: unknown): unknown {
  if (Buffer.isBuffer(v)) return v.toString('base64');
  if (v instanceof Uint8Array) return Buffer.from(v).toString('base64');
  return v;
}

function normalizeRows(rows: unknown[]): Record<string, unknown>[] {
  return rows.map((row) => {
    const out: Record<string, unknown> = {};
    if (typeof row === 'object' && row !== null) {
      const r = row as Record<string, unknown>;
      for (const key of Object.keys(r)) out[key] = normalizeRowValue(r[key]);
    }
    return out;
  });
}

type TrackedAdapter = {
  adapter: StorageAdapter;
  lastUsed: number;
  isBusy: boolean;
  // Held from `begin` until `commit`/`rollback`/`close` to serialize SQLite transactions.
  releaseSqliteLock?: () => void;
};

export class StorageManager {
  private readonly factory: ConnFactory;
  /** Single adapter used only to detect the SQL dialect at construction time. */
  private readonly dialectAdapter: StorageAdapter;
  private readonly dialectOverride?: string;
  private readonly connections = new Map<number, TrackedAdapter>();
  private readonly inFlight = new Set<Promise<void>>();
  private nextConnId = 1;
  // SQLite only: promise chain that serializes begin→commit/rollback sequences so
  // two concurrent write_batch calls can't interleave transactions on the same DB object.
  private sqliteQueue: Promise<void> = Promise.resolve();
  private engineShutdown?: () => void;
  private engineBuild?: () => Promise<void>;
  private sweepHandle?: ReturnType<typeof setInterval>;

  constructor(factory: ConnFactory, dialectOverride?: string) {
    this.factory = factory;
    this.dialectOverride = dialectOverride;
    this.dialectAdapter = Registry.getAdapter(factory);
    const handle = setInterval(() => {
      this.sweepOrphanedConnections();
    }, SWEEP_INTERVAL_MS);
    handle.unref();
    this.sweepHandle = handle;
  }

  public getDialect(): string {
    return this.dialectOverride ?? this.dialectAdapter.getDialect();
  }

  public setEngineShutdown(fn: () => void): void {
    this.engineShutdown = fn;
  }

  public setEngineBuild(fn: () => Promise<void>): void {
    this.engineBuild = fn;
  }

  /** Runs database migrations. Call once after construction, before any SDK operations. */
  public async build(): Promise<void> {
    if (this.engineBuild) {
      await this.engineBuild();
    }
  }

  /**
   * Entry point for every Rust storage call. Parses the JSON payload, dispatches
   * to the right operation, then calls `resolve` with the JSON result.
   *
   * `resolve` must be called exactly once — it unblocks the waiting Rust thread.
   */
  public handleStorageCall(
    id: number,
    payloadJson: string,
    resolve: (result: object) => void
  ): void {
    let payload: StorageCallPayload;
    try {
      payload = JSON.parse(payloadJson) as StorageCallPayload;
    } catch {
      resolve({ error: { code: 'JSON_ERR', message: 'invalid JSON from Rust' } });
      return;
    }

    const p = this.dispatchOp(id, payload, resolve).catch((e: unknown) => {
      const code = typeof e === 'object' && e !== null && 'code' in e ? String(e.code) : 'ERR';
      resolve({ error: { code, message: String(e) } });
    });
    this.inFlight.add(p);
    void p.finally(() => this.inFlight.delete(p));
  }

  /**
   * Releases connections that Rust acquired but never closed — e.g. after a panic
   * mid-sequence that bypassed the normal `{ op: "close" }` message. Called on
   * every `acquire` so orphan cleanup is driven by natural activity with no timer.
   */
  private sweepOrphanedConnections(): void {
    const cutoff = Date.now() - CONN_TTL_MS;
    for (const [id, entry] of this.connections) {
      if (!entry.isBusy && entry.lastUsed < cutoff) {
        this.connections.delete(id);
        entry.releaseSqliteLock?.();
        entry.releaseSqliteLock = undefined;
        const p = Promise.resolve(entry.adapter.close()).catch((e: unknown) => {
          console.warn(`[Memori] failed to close orphaned connection ${id}:`, e);
        });
        this.inFlight.add(p);
        void p.finally(() => this.inFlight.delete(p));
      }
    }
  }

  private async dispatchOp(
    _id: number,
    payload: StorageCallPayload,
    resolve: (result: object) => void
  ): Promise<void> {
    switch (payload.op) {
      case 'acquire': {
        this.sweepOrphanedConnections();
        const adapter = Registry.getAdapter(this.factory);
        const connId = this.nextConnId++;
        this.connections.set(connId, { adapter, lastUsed: Date.now(), isBusy: false });
        resolve({ conn_id: connId });
        break;
      }

      case 'execute': {
        const connId = payload.conn_id ?? -1;
        const entry = this.connections.get(connId);
        if (!entry) {
          resolve({ error: { code: 'NO_CONN', message: `unknown conn_id: ${connId}` } });
          return;
        }
        entry.lastUsed = Date.now();
        entry.isBusy = true;
        try {
          const binds = deserializeBinds(payload.binds ?? []);
          const rawRows = await entry.adapter.execute(payload.sql ?? '', binds);
          resolve({ rows: normalizeRows(rawRows) });
        } finally {
          entry.isBusy = false;
          entry.lastUsed = Date.now();
        }
        break;
      }

      case 'begin': {
        const connId = payload.conn_id ?? -1;
        const entry = this.connections.get(connId);
        if (!entry) {
          resolve({ error: { code: 'NO_CONN', message: `unknown conn_id: ${connId}` } });
          return;
        }
        // SQLite: wait for any in-progress transaction to finish before starting a new one,
        // mirroring Python's connection_context which scopes one connection per operation.
        if (this.getDialect() === 'sqlite') {
          const prev = this.sqliteQueue;
          let releaseLock!: () => void;
          this.sqliteQueue = new Promise<void>((r) => {
            releaseLock = r;
          });
          await prev;
          entry.releaseSqliteLock = releaseLock;
        }
        entry.lastUsed = Date.now();
        entry.isBusy = true;
        let began = false;
        try {
          await entry.adapter.begin();
          began = true;
          resolve({ ok: true });
        } finally {
          if (!began) {
            entry.releaseSqliteLock?.();
            entry.releaseSqliteLock = undefined;
          }
          entry.isBusy = false;
          entry.lastUsed = Date.now();
        }
        break;
      }

      case 'commit': {
        const connId = payload.conn_id ?? -1;
        const entry = this.connections.get(connId);
        if (!entry) {
          resolve({ error: { code: 'NO_CONN', message: `unknown conn_id: ${connId}` } });
          return;
        }
        entry.lastUsed = Date.now();
        entry.isBusy = true;
        try {
          await entry.adapter.commit();
          resolve({ ok: true });
        } finally {
          entry.releaseSqliteLock?.();
          entry.releaseSqliteLock = undefined;
          entry.isBusy = false;
          entry.lastUsed = Date.now();
        }
        break;
      }

      case 'rollback': {
        const connId = payload.conn_id ?? -1;
        const entry = this.connections.get(connId);
        if (!entry) {
          // Rollback failure is non-fatal — connection may already be gone.
          resolve({ ok: true });
          return;
        }
        entry.lastUsed = Date.now();
        entry.isBusy = true;
        try {
          await entry.adapter.rollback();
        } catch {
          // non-fatal
        } finally {
          entry.releaseSqliteLock?.();
          entry.releaseSqliteLock = undefined;
          entry.isBusy = false;
          entry.lastUsed = Date.now();
        }
        resolve({ ok: true });
        break;
      }

      case 'close': {
        const connId = payload.conn_id ?? -1;
        const entry = this.connections.get(connId);
        this.connections.delete(connId);
        if (entry) {
          entry.releaseSqliteLock?.();
          entry.releaseSqliteLock = undefined;
          try {
            await entry.adapter.close();
          } catch {
            // non-fatal
          }
        }
        resolve({ ok: true });
        break;
      }

      default:
        resolve({ error: { code: 'UNKNOWN_OP', message: `unknown op: ${payload.op}` } });
    }
  }

  public async close(): Promise<void> {
    clearInterval(this.sweepHandle);
    this.sweepHandle = undefined;
    if (this.engineShutdown) {
      this.engineShutdown();
      this.engineShutdown = undefined;
    }
    // Drain all in-flight dispatchOp calls before touching adapters.
    await Promise.allSettled(this.inFlight);
    // Release any connections that Rust left open (e.g. due to an in-flight shutdown).
    for (const entry of this.connections.values()) {
      entry.releaseSqliteLock?.();
      try {
        await entry.adapter.close();
      } catch {
        // non-fatal
      }
    }
    this.connections.clear();
    try {
      await this.dialectAdapter.close();
    } catch {
      // non-fatal
    }
  }
}
