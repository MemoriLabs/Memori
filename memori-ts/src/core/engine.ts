import { MemoriEngine } from '../native/index.js';
import {
  StorageBridge,
  WriteBatch,
  EmbeddingRow,
  CandidateFactRow,
  WriteAck,
} from '../types/storage.js';
import { RetrievalRequest, RecallObject, NapiRecallRow } from '../types/api.js';
import { AugmentationInput } from '../types/integrations.js';

type BridgeCb = (err: Error | null, _id: number, _reqJson: string) => void;

export class NativeEngine {
  private memoriEngine?: MemoriEngine;
  private _hasStorage: boolean = false;
  private readonly modelName: string | null;
  private readonly fetchEmbeddingsCb: BridgeCb;
  private readonly fetchFactsCb: BridgeCb;
  private readonly writeBatchCb: BridgeCb;

  constructor(storageBridge?: StorageBridge, modelName?: string) {
    this.modelName = modelName ?? null;

    if (storageBridge) {
      this._hasStorage = true;

      this.fetchEmbeddingsCb = (err, _id, _reqJson) => {
        if (err) {
          console.error('[Memori] Bridge error in fetchEmbeddings:', err);
          return;
        }
        const [id, reqJson] = (Array.isArray(_id) ? _id : [_id, _reqJson]) as [number, string];
        this.handleBridgeCall<EmbeddingRow[]>(
          id,
          'fetchEmbeddings',
          () => {
            const req = JSON.parse(reqJson) as { entity_id: string; limit: number };
            return storageBridge.fetchEmbeddings(req.entity_id, req.limit);
          },
          (res) =>
            this.memoriEngine?.resolveEmbeddingsCallback(
              id,
              res.map((r) => ({
                id: r.id,
                contentEmbedding: r.content_embedding ?? new Float32Array(0),
              }))
            ),
          () => this.memoriEngine?.resolveEmbeddingsCallback(id, [])
        );
      };

      this.fetchFactsCb = (err, _id, _reqJson) => {
        if (err) {
          console.error('[Memori] Bridge error in fetchFactsByIds:', err);
          return;
        }
        const [id, reqJson] = (Array.isArray(_id) ? _id : [_id, _reqJson]) as [number, string];
        this.handleBridgeCall<CandidateFactRow[]>(
          id,
          'fetchFactsByIds',
          () => {
            const req = JSON.parse(reqJson) as { ids: (number | string)[] };
            return storageBridge.fetchFactsByIds(req.ids);
          },
          (res) =>
            this.memoriEngine?.resolveFactsCallback(
              id,
              res.map((r) => ({
                id: r.id,
                content: r.content,
                dateCreated: r.date_created,
                summaries: r.summaries?.map((s) => ({
                  content: s.content,
                  dateCreated: s.date_created,
                })),
              }))
            ),
          () => this.memoriEngine?.resolveFactsCallback(id, [])
        );
      };

      this.writeBatchCb = (err, _id, _reqJson) => {
        if (err) {
          console.error('[Memori] Bridge error in writeBatch:', err);
          return;
        }
        const [id, reqJson] = (Array.isArray(_id) ? _id : [_id, _reqJson]) as [number, string];
        this.handleBridgeCall<WriteAck>(
          id,
          'writeBatch',
          () => {
            const req = JSON.parse(reqJson) as WriteBatch;
            return storageBridge.writeBatch(req);
          },
          (res) => this.memoriEngine?.resolveWriteCallback(id, { writtenOps: res.written_ops }),
          () => this.memoriEngine?.resolveWriteCallback(id, { writtenOps: 0 })
        );
      };
    } else {
      this.fetchEmbeddingsCb = (err, _id, _reqJson) => {
        const [id] = (Array.isArray(_id) ? _id : [_id, _reqJson]) as [number, string];
        if (!err) this.memoriEngine?.resolveEmbeddingsCallback(id, []);
      };
      this.fetchFactsCb = (err, _id, _reqJson) => {
        const [id] = (Array.isArray(_id) ? _id : [_id, _reqJson]) as [number, string];
        if (!err) this.memoriEngine?.resolveFactsCallback(id, []);
      };
      this.writeBatchCb = (err, _id, _reqJson) => {
        const [id] = (Array.isArray(_id) ? _id : [_id, _reqJson]) as [number, string];
        if (!err) this.memoriEngine?.resolveWriteCallback(id, { writtenOps: 0 });
      };
    }
  }

  private getEngine(): MemoriEngine {
    if (!this.memoriEngine) {
      this.memoriEngine = new MemoriEngine(
        this.modelName,
        this.fetchEmbeddingsCb,
        this.fetchFactsCb,
        this.writeBatchCb
      );
    }
    return this.memoriEngine;
  }

  public get hasStorage(): boolean {
    return this._hasStorage;
  }

  /**
   * Helper to execute storage bridge callbacks safely, handling both async Promises
   * and sync returns, while catching all crossing boundary errors.
   */
  private handleBridgeCall<TRes>(
    id: number,
    operationName: string,
    executeFn: () => Promise<TRes> | TRes,
    successCb: (res: TRes) => void,
    fallbackCb: () => void
  ) {
    try {
      const result = executeFn();

      if (result instanceof Promise) {
        result
          .then((res) => {
            try {
              successCb(res);
            } catch (e: unknown) {
              console.error(`[Memori] Bridge Error in ${operationName} (success handler):`, e);
              fallbackCb();
            }
          })
          .catch((err: unknown) => {
            console.error(`[Memori] Bridge Error in ${operationName}:`, err);
            fallbackCb();
          });
      } else {
        successCb(result);
      }
    } catch (e: unknown) {
      console.error(`[Memori] Bridge Sync Error (${operationName}):`, e);
      fallbackCb();
    }
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
        entity_fact_id: s.entityFactId as number,
        fact_id: s.factId as number,
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

  public embedTexts(texts: string[]): Float32Array[] {
    if (texts.length === 0) return [];
    try {
      return this.getEngine().embedTexts(texts);
    } catch (e: unknown) {
      console.error('[Memori] Bridge Sync Error (embedTexts):', e);
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
    return await this.memoriEngine.waitForAugmentation(timeoutMs);
  }

  public shutdown(): void {
    if (!this.memoriEngine) return;
    if (typeof this.memoriEngine.shutdown === 'function') {
      this.memoriEngine.shutdown();
    }
    this.memoriEngine = undefined;
    this._hasStorage = false;
  }
}
