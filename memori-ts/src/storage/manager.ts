import { StorageAdapter, BaseDriver, ConnFactory } from './base.js';
import { Registry } from './registry.js';
import { Builder } from './builder.js';
import { Config } from '../core/config.js';
import {
  StorageBridge,
  WriteBatch,
  WriteAck,
  CandidateFactRow,
  EmbeddingRow,
} from '../types/storage.js';

// Side-effect imports: each module calls Registry.registerAdapter / Registry.registerDriver
// on load, so the Registry can auto-detect the connection type at runtime.
import './adapters/postgresql.js';
import './drivers/postgresql.js';
import './adapters/sqlite.js';
import './drivers/sqlite.js';
import './adapters/mysql.js';
import './drivers/mysql.js';

/**
 * The host-side implementation of `StorageBridge`.
 *
 * Sits between the Rust core (which calls `fetchEmbeddings`, `fetchFactsByIds`, `writeBatch`)
 * and the user's database connection. Auto-detects the ORM/driver from the raw connection
 * object via the `Registry` and delegates all SQL to the appropriate `BaseDriver`.
 */
export class StorageManager implements StorageBridge {
  private readonly factory: ConnFactory;
  private readonly adapter: StorageAdapter;
  private readonly driver: BaseDriver;
  private readonly config: Config;
  private embedder?: (texts: string[]) => Float32Array[];
  private engineShutdown?: () => void;

  constructor(factory: ConnFactory) {
    this.factory = factory;
    this.config = new Config();
    this.adapter = Registry.getAdapter(factory);
    this.driver = Registry.getDriver(this.adapter);
  }

  /**
   * Wires the native engine's embed function into the storage manager.
   * Called after construction because the engine is built with a reference to this manager,
   * so we can't pass the embedder in the constructor without a circular dependency.
   */
  public setEmbedder(fn: (texts: string[]) => Float32Array[]): void {
    this.embedder = fn;
  }

  public setEngineShutdown(fn: () => void): void {
    this.engineShutdown = fn;
  }

  public async build(): Promise<void> {
    const builder = new Builder(this.config, this.adapter, this.driver);
    await builder.execute();
  }

  public async close(): Promise<void> {
    if (this.engineShutdown) {
      this.engineShutdown();
      this.engineShutdown = undefined;
    }
    // Brief delay to let any in-flight async write callbacks settle before
    // closing the underlying connection (e.g. SQLite WAL checkpointing).
    await new Promise((resolve) => setTimeout(resolve, 100));
    await this.adapter.close();
  }

  public getDialect(): string {
    return this.adapter.getDialect();
  }

  public async fetchEmbeddings(entityId: string, limit: number): Promise<EmbeddingRow[]> {
    const eId = await this.driver.entity.create(entityId);
    const rows = await this.driver.entityFact.getEmbeddings(eId || entityId, limit);
    return rows;
  }

  public async fetchFactsByIds(ids: (number | string)[]): Promise<CandidateFactRow[]> {
    return await this.driver.entityFact.getFactsByIds(ids);
  }

  public async getConversationHistory(
    sessionId: string
  ): Promise<Array<{ role: string; content: string }>> {
    const sId = await this.driver.session.create(sessionId, null, null);
    const convId = await this.driver.conversation.create(sId || sessionId, 30);
    const internalConvId = convId || sessionId;

    return await this.driver.conversationMessages.read(internalConvId);
  }

  public writeBatch(batch: WriteBatch): Promise<WriteAck> {
    return this.writeBatchAsync(batch);
  }

  /**
   * Executes all write operations from an augmentation batch inside a single transaction.
   *
   * Each `op_type` maps to a specific set of driver calls. The entire batch rolls back on
   * any failure — `written_ops: 0` is returned rather than throwing so the Rust caller
   * (which called this via the storage bridge) gets a clean, non-panicking result.
   */
  private async writeBatchAsync(batch: WriteBatch): Promise<WriteAck> {
    if (batch.ops.length === 0) return { written_ops: 0 };

    // CockroachDB (and PostgreSQL with SERIALIZABLE) can return code 40001 when a
    // transaction conflicts with a recent commit. The correct response is to retry
    // the entire transaction from scratch with a fresh connection.
    const MAX_RETRIES = 5;
    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      let written = 0;

      // Each write batch gets its own adapter+driver pair so its dedicated
      // transaction connection never races with concurrent reads on this.adapter.
      const txAdapter = Registry.getAdapter(this.factory);
      const txDriver = Registry.getDriver(txAdapter);

      try {
        await txAdapter.begin();

        for (const op of batch.ops) {
          switch (op.op_type) {
            case 'conversation_message.create': {
              const sId = await txDriver.session.create(op.payload.conversation_id, null, null);
              const convId = await txDriver.conversation.create(
                sId || op.payload.conversation_id,
                30
              );
              const internalConvId = convId || op.payload.conversation_id;
              for (const message of op.payload.messages) {
                await txDriver.conversationMessage.create(
                  internalConvId,
                  message.role,
                  'text',
                  message.content
                );
              }
              break;
            }
            case 'entity_fact.create': {
              const eId = await txDriver.entity.create(op.payload.entity_id);
              // driver.entity.create uses INSERT OR IGNORE and returns the internal row ID;
              // fall back to external_id if the row pre-existed and no ID was returned
              const internalEntityId = eId || op.payload.entity_id;
              let factEmbeddings = op.payload.fact_embeddings;
              if (
                (!factEmbeddings || factEmbeddings.length === 0) &&
                this.embedder &&
                op.payload.facts.length > 0
              ) {
                factEmbeddings = this.embedder(op.payload.facts);
              }
              let internalConvId = null;
              if (op.payload.conversation_id) {
                const sId = await txDriver.session.create(
                  op.payload.conversation_id,
                  internalEntityId,
                  null
                );
                // 30 = conversation inactivity timeout in minutes before a new one is created
                internalConvId = await txDriver.conversation.create(
                  sId || op.payload.conversation_id,
                  30
                );
              }
              await txDriver.entityFact.create(
                internalEntityId,
                op.payload.facts,
                factEmbeddings,
                internalConvId
              );
              break;
            }
            case 'knowledge_graph.create': {
              const eId = await txDriver.entity.create(op.payload.entity_id);
              await txDriver.knowledgeGraph.create(
                eId || op.payload.entity_id,
                op.payload.semantic_triples
              );
              break;
            }
            case 'process_attribute.create': {
              const pId = await txDriver.process.create(op.payload.process_id);
              await txDriver.processAttribute.create(
                pId || op.payload.process_id,
                Array.isArray(op.payload.attributes)
                  ? op.payload.attributes
                  : Object.values(op.payload.attributes)
              );
              break;
            }
            case 'conversation.update': {
              const sId = await txDriver.session.create(op.payload.conversation_id, null, null);
              const convId = await txDriver.conversation.create(
                sId || op.payload.conversation_id,
                30
              );
              await txDriver.conversation.update(
                convId || op.payload.conversation_id,
                op.payload.summary
              );
              break;
            }
            case 'upsert_fact': {
              const eId = await txDriver.entity.create(op.payload.entity_id);
              if (op.payload.content)
                await txDriver.entityFact.createWithoutEmbedding(
                  eId || op.payload.entity_id,
                  op.payload.content
                );
              break;
            }
            default: {
              throw new Error(`Unsupported write op type: ${(op as { op_type: string }).op_type}`);
            }
          }
          written++;
        }

        await txAdapter.commit();
        return { written_ops: written };
      } catch (e) {
        try {
          await txAdapter.rollback();
        } catch {
          // rollback failure is non-fatal
        }
        if ((e as { code?: string }).code === '40001' && attempt < MAX_RETRIES) {
          await new Promise((r) => setTimeout(r, Math.min(50 * 2 ** attempt, 1000)));
          continue;
        }
        console.error(`[Memori] Async WriteBatch failed, rolling back:`, e);
        return { written_ops: 0 };
      }
    }
    return { written_ops: 0 };
  }
}
