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

// Binary embeddings arrive as base64 strings and are decoded to Buffer; ints arrive as strings (CockroachDB i64).
function deserializeBinds(binds: Array<{ t: string; v: unknown }>): SqlBindValue[] {
  return binds.map((bind) => {
    switch (bind.t) {
      case 'null':
        return null;
      case 'int': // string: CockroachDB returns i64s that exceed JS Number.MAX_SAFE_INTEGER
      case 'text':
        return bind.v as string;
      case 'float':
        return bind.v as number;
      case 'bytes':
        return Buffer.from(bind.v as string, 'base64');
      default:
        console.warn(`[Memori] unknown bind type "${bind.t}" — treating as NULL`);
        return null;
    }
  });
}

const CONN_TTL_MS = 30_000;
const STALE_TX_TTL_MS = 60_000;
const SWEEP_INTERVAL_MS = 60_000;

function normalizeRowValue(v: unknown): unknown {
  if (Buffer.isBuffer(v)) return v.toString('base64');
  if (v instanceof Uint8Array) return Buffer.from(v).toString('base64');
  if (typeof v === 'bigint') return v.toString();
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
  inTransaction: boolean;
  // Held from acquire until close to serialize access for adapters that require it (SQLite, MySQL direct).
  releaseSerialLock?: () => void;
};

export class StorageManager {
  private readonly factory: ConnFactory;
  private readonly cachedDialect: string;
  private readonly dialectOverride?: string;
  private readonly connections = new Map<number, TrackedAdapter>();
  private readonly inFlight = new Set<Promise<void>>();
  private nextConnId = 1;
  // Serializes acquire→close for adapters that require it (SQLite, MySQL direct).
  private serialQueue: Promise<void> = Promise.resolve();
  private readonly needsSerialAccess: boolean;
  private engineShutdown?: () => void;
  private engineBuild?: () => Promise<void>;
  private sweepHandle?: ReturnType<typeof setInterval>;

  constructor(factory: ConnFactory, dialectOverride?: string) {
    this.factory = factory;
    this.dialectOverride = dialectOverride;
    const probe = Registry.getAdapter(factory);
    this.cachedDialect = probe.getDialect();
    this.needsSerialAccess = probe.requiresSerialAccess?.() ?? false;
    void Promise.resolve(probe.close()).catch(() => {});
    const handle = setInterval(() => {
      this.sweepOrphanedConnections();
    }, SWEEP_INTERVAL_MS);
    handle.unref();
    this.sweepHandle = handle;
  }

  public getDialect(): string {
    return this.dialectOverride ?? this.cachedDialect;
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

  // `resolve` must be called exactly once — it unblocks the waiting Rust thread.
  public handleStorageCall(
    _id: number,
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

    const p = this.dispatchOp(payload, resolve).catch((e: unknown) => {
      // Prefer sqlState over code: mysql2 reports serialization failures (e.g. deadlock)
      // as sqlState "40001" while code is a symbolic name like ER_LOCK_DEADLOCK. Rust's
      // retry logic matches on the numeric SQLSTATE, so normalize here for all dialects.
      let code = 'ERR';
      if (typeof e === 'object' && e !== null) {
        const err = e as Record<string, unknown>;
        const raw = err['sqlState'] ?? err['code'];
        if (typeof raw === 'string' || typeof raw === 'number') code = String(raw);
      }
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
    const txCutoff = Date.now() - STALE_TX_TTL_MS;
    for (const [id, entry] of this.connections) {
      const isStaleIdle = !entry.isBusy && !entry.inTransaction && entry.lastUsed < cutoff;
      const isStaleTx = !entry.isBusy && entry.inTransaction && entry.lastUsed < txCutoff;
      if (!isStaleIdle && !isStaleTx) continue;

      this.connections.delete(id);
      entry.releaseSerialLock?.();
      entry.releaseSerialLock = undefined;

      const p = isStaleTx
        ? Promise.resolve(entry.adapter.rollback())
            .catch(() => {})
            .then(() => entry.adapter.close())
            .catch((e: unknown) => {
              console.warn(`[Memori] failed to close orphaned transaction ${id}:`, e);
            })
        : Promise.resolve(entry.adapter.close()).catch((e: unknown) => {
            console.warn(`[Memori] failed to close orphaned connection ${id}:`, e);
          });
      this.inFlight.add(p);
      void p.finally(() => this.inFlight.delete(p));
    }
  }

  private requireEntry(
    connId: number | undefined,
    resolve: (result: object) => void
  ): TrackedAdapter | undefined {
    const entry = this.connections.get(connId ?? -1);
    if (!entry) resolve({ error: { code: 'NO_CONN', message: `unknown conn_id: ${connId}` } });
    return entry;
  }

  private async dispatchOp(
    payload: StorageCallPayload,
    resolve: (result: object) => void
  ): Promise<void> {
    switch (payload.op) {
      case 'acquire': {
        this.sweepOrphanedConnections();
        const adapter = Registry.getAdapter(this.factory);
        const connId = this.nextConnId++;

        let releaseSerialLock: (() => void) | undefined;
        if (this.needsSerialAccess) {
          const prev = this.serialQueue;
          let releaseLock!: () => void;
          this.serialQueue = new Promise<void>((r) => {
            releaseLock = r;
          });
          await prev;
          if (this.sweepHandle === undefined) {
            releaseLock();
            resolve({
              error: { code: 'SHUTTING_DOWN', message: 'storage manager is shutting down' },
            });
            return;
          }
          releaseSerialLock = releaseLock;
        }

        this.connections.set(connId, {
          adapter,
          lastUsed: Date.now(),
          isBusy: false,
          inTransaction: false,
          releaseSerialLock,
        });
        resolve({ conn_id: connId });
        break;
      }

      case 'execute': {
        const entry = this.requireEntry(payload.conn_id, resolve);
        if (!entry) return;
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
        const entry = this.requireEntry(payload.conn_id, resolve);
        if (!entry) return;
        // Serial-access serialization is enforced at acquire time (lock spans acquire→close),
        // so we only need a shutdown guard here in case close() was called after acquire.
        if (this.needsSerialAccess && this.sweepHandle === undefined) {
          resolve({
            error: { code: 'SHUTTING_DOWN', message: 'storage manager is shutting down' },
          });
          return;
        }
        entry.lastUsed = Date.now();
        entry.isBusy = true;
        let began = false;
        try {
          await entry.adapter.begin();
          began = true;
          entry.inTransaction = true;
          resolve({ ok: true });
        } finally {
          if (!began) {
            // begin failed — release the lock so the queue doesn't get stuck waiting for a close that may not come.
            entry.releaseSerialLock?.();
            entry.releaseSerialLock = undefined;
          }
          entry.isBusy = false;
          entry.lastUsed = Date.now();
        }
        break;
      }

      case 'commit': {
        const entry = this.requireEntry(payload.conn_id, resolve);
        if (!entry) return;
        entry.lastUsed = Date.now();
        entry.isBusy = true;
        try {
          await entry.adapter.commit();
          resolve({ ok: true });
        } finally {
          entry.inTransaction = false;
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
          entry.inTransaction = false;
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
          try {
            if (entry.inTransaction) {
              await entry.adapter.rollback();
            }
            await entry.adapter.close();
          } catch {
            // non-fatal
          } finally {
            // Release the SQLite lock only after rollback/close so the next acquire
            // doesn't start on the shared sqlite3* handle before cleanup finishes.
            entry.releaseSerialLock?.();
            entry.releaseSerialLock = undefined;
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
    // Signal shutdown and release all held SQLite locks before draining. Any begin
    // calls queued behind an open transaction will see isShuttingDown, call their own
    // releaseLock (unwinding the full chain), and resolve with a SHUTTING_DOWN error
    // rather than hanging forever waiting for a commit that will never arrive.
    for (const entry of this.connections.values()) {
      entry.releaseSerialLock?.();
      entry.releaseSerialLock = undefined;
    }
    // Drain all in-flight dispatchOp calls before touching adapters.
    await Promise.allSettled(this.inFlight);
    // Release any connections that Rust left open (e.g. due to an in-flight shutdown).
    // Roll back any open transactions first so the DB isn't left in a dirty state.
    for (const entry of this.connections.values()) {
      if (entry.inTransaction) {
        try {
          await entry.adapter.rollback();
        } catch {
          // non-fatal
        }
      }
      try {
        await entry.adapter.close();
      } catch {
        // non-fatal
      }
    }
    this.connections.clear();
  }
}
