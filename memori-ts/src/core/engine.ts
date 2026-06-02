import type { MemoriEngine } from '../native/index.js';
import { createRequire } from 'node:module';
import { WriteBatch, WriteAck } from '../types/storage.js';
import { RetrievalRequest, RecallObject, NapiRecallRow } from '../types/api.js';
import { AugmentationInput } from '../types/integrations.js';
import { StorageManager } from '../storage/manager.js';

type StorageCallCb = (
  err: Error | null,
  _id: number | [number, string],
  _payloadJson?: string
) => void;

export class NativeEngine {
  private memoriEngine?: MemoriEngine;
  private _hasStorage: boolean = false;
  private _isShutdown: boolean = false;
  private readonly modelName: string | null;
  private readonly storageManager?: StorageManager;
  // One process.once('beforeExit') shared across all instances — prevents
  // MaxListenersExceededWarning regardless of how many engines are created.
  private static readonly _activeEngines = new Set<NativeEngine>();
  private static _beforeExitInstalled = false;

  constructor(storageManager?: StorageManager, modelName?: string) {
    this.modelName = modelName ?? null;
    this.storageManager = storageManager;
    if (storageManager) this._hasStorage = true;
  }

  private static _installBeforeExitOnce(): void {
    if (NativeEngine._beforeExitInstalled) return;
    NativeEngine._beforeExitInstalled = true;
    process.once('beforeExit', () => {
      for (const engine of Array.from(NativeEngine._activeEngines)) {
        engine.shutdown();
      }
    });
  }

  private getEngine(): MemoriEngine {
    if (this._isShutdown) throw new Error('[Memori] engine has been shut down');
    if (!this.memoriEngine) {
      const require = createRequire(import.meta.url);
      const native = require('../native/index.js') as {
        MemoriEngine: new (
          modelName: string | null,
          storageCallCb: StorageCallCb,
          dialect: string
        ) => MemoriEngine;
      };

      if (this.storageManager) {
        const dialect = this.storageManager.getDialect();
        const sm = this.storageManager;
        const storageCallCb: StorageCallCb = (err, _id, _payloadJson) => {
          const id = Array.isArray(_id) ? _id[0] : _id;
          if (err) {
            console.error('[Memori] Bridge error in storageCall:', err);
            this.memoriEngine?.resolveStorageCall(
              id,
              JSON.stringify({ error: { code: 'NAPI_ERR', message: err.message } })
            );
            return;
          }
          const [, payloadJson] = Array.isArray(_id) ? _id : [_id, _payloadJson as string];
          sm.handleStorageCall(id, payloadJson, (result) => {
            this.memoriEngine?.resolveStorageCall(id, JSON.stringify(result));
          });
        };
        this.memoriEngine = new native.MemoriEngine(this.modelName, storageCallCb, dialect);
      } else {
        // No storage configured — provide a no-op callback so Rust doesn't hang.
        const noopCb: StorageCallCb = (err, _id) => {
          if (err) return;
          const id = Array.isArray(_id) ? _id[0] : _id;
          this.memoriEngine?.resolveStorageCall(
            id,
            JSON.stringify({ error: { code: 'NO_STORAGE', message: 'no storage configured' } })
          );
        };
        this.memoriEngine = new native.MemoriEngine(this.modelName, noopCb, 'sqlite');
      }
      // Safety net: track this engine so the shared beforeExit listener can
      // clean up tokio WorkerRuntime threads when the event loop drains.
      NativeEngine._activeEngines.add(this);
      NativeEngine._installBeforeExitOnce();
    }
    return this.memoriEngine;
  }

  public get hasStorage(): boolean {
    return this._hasStorage;
  }

  /** Runs database migrations. Must be called once after the engine is constructed with storage. */
  public async build(): Promise<void> {
    if (this._hasStorage) {
      await this.getEngine().build();
    }
  }

  /**
   * Executes a write batch through the Rust storage layer.
   *
   * Used by the TS persistence engine for immediate conversation message writes
   * before the async augmentation pipeline completes.
   */
  public async writeBatch(batch: WriteBatch): Promise<WriteAck> {
    if (!this._hasStorage) return { written_ops: 0 };
    const ack = await this.getEngine().writeBatch(JSON.stringify(batch));
    return { written_ops: ack.writtenOps };
  }

  /**
   * Returns the conversation history for a session from the Rust storage layer.
   */
  public async getConversationHistory(
    sessionId: string
  ): Promise<Array<{ role: string; content: string }>> {
    if (!this._hasStorage) return [];
    const json = await this.getEngine().getConversationHistory(sessionId);
    return JSON.parse(json) as Array<{ role: string; content: string }>;
  }

  public async retrieve(request: RetrievalRequest): Promise<RecallObject[]> {
    const napiResults: NapiRecallRow[] = await this.getEngine().retrieve({
      entityId: request.entity_id,
      queryText: request.query_text,
      denseLimit: request.dense_limit,
      limit: request.limit,
    });

    // Map N-API camelCase back to TS snake_case
    return napiResults.map((r) => ({
      id: r.id,
      content: r.content,
      rank_score: r.rankScore ?? undefined,
      similarity: r.similarity ?? undefined,
      date_created: r.dateCreated ?? undefined,
      summaries: r.summaries?.map((s) => ({
        content: s.content,
        date_created: s.dateCreated,
        entity_fact_id: s.entityFactId as number | string,
        fact_id: s.factId as number | string,
      })),
    }));
  }

  public async recall(request: RetrievalRequest): Promise<string> {
    return await this.getEngine().recall({
      entityId: request.entity_id,
      queryText: request.query_text,
      denseLimit: request.dense_limit,
      limit: request.limit,
    });
  }

  public async embedTexts(texts: string[]): Promise<Float32Array[]> {
    if (texts.length === 0) return [];
    try {
      return await this.getEngine().embedTexts(texts);
    } catch (e: unknown) {
      console.error('[Memori] Bridge Error (embedTexts):', e);
      return [];
    }
  }

  public submitAugmentation(input: AugmentationInput): string {
    return this.getEngine().submitAugmentation({
      entityId: input.entity_id,
      processId: input.process_id ?? undefined,
      conversationId: input.conversation_id ?? undefined,
      conversationMessages: input.conversation_messages ?? undefined,
      systemPrompt: input.system_prompt ?? undefined,
      llmProvider: input.llm_provider ?? undefined,
      llmModel: input.llm_model ?? undefined,
      llmProviderSdkVersion: input.llm_provider_sdk_version ?? undefined,
      framework: input.framework ?? undefined,
      platformProvider: input.platform_provider ?? undefined,
      storageDialect: input.storage_dialect ?? undefined,
      storageCockroachdb: input.storage_cockroachdb ?? undefined,
      sdkVersion: input.sdk_version ?? undefined,
      useMockResponse: input.use_mock_response ?? undefined,
      sessionId: input.session_id ?? undefined,
      factId: input.fact_id ?? undefined,
      content: input.content ?? undefined,
    });
  }

  public async waitForAugmentation(timeoutMs?: number): Promise<boolean> {
    if (!this.memoriEngine) return false;
    const result = await this.memoriEngine.waitForAugmentation(timeoutMs);
    // Rust's NodeConnection::close() is fire-and-forget (NonBlocking TSFN). By the
    // time waitForAugmentation resolves, close messages are queued but not yet
    // dispatched. One event-loop tick lets the TS side process them so pool
    // connections are fully released before the caller hits pool.end().
    await new Promise<void>((resolve) => setImmediate(resolve));
    return result;
  }

  public shutdown(): void {
    NativeEngine._activeEngines.delete(this);
    this._hasStorage = false;
    this._isShutdown = true;
    if (this.memoriEngine) {
      if (typeof this.memoriEngine.shutdown === 'function') {
        this.memoriEngine.shutdown();
      }
      this.memoriEngine = undefined;
    }
  }
}
