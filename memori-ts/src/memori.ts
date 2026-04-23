import { Axon } from '@memorilabs/axon';
import { Config } from './core/config.js';
import { SessionManager } from './core/session.js';
import { ProjectManager } from './core/project.js';
import { Api, ApiSubdomain } from './core/network.js';
import { RecallEngine } from './engines/recall.js';
import { PersistenceEngine } from './engines/persistence.js';
import { AugmentationEngine } from './engines/augmentation.js';
import { ParsedFact } from './types/api.js';
import { IntegrationConstructor, SupportedIntegration } from './types/integrations.js';
import { NativeEngine } from './core/engine.js';
import { StorageManager } from './storage/manager.js';

export interface MemoriOptions {
  conn?: unknown; // The raw database connection (pg Pool, mysql2 conn, better-sqlite3 DB, etc.)
  embeddingModel?: string;
}

/**
 * The main entry point for the Memori SDK.
 *
 * This class orchestrates the connection between your application, the Memori Cloud,
 * and your LLM provider. It automatically handles:
 * - Long-term memory recall (fetching relevant facts)
 * - Conversation persistence (storing messages)
 * - User augmentation (learning from interactions)
 */
export class Memori {
  /**
   * The configuration state for the SDK.
   * Modifying properties here (like timeout) affects all future requests.
   */
  public readonly config: Config;

  /**
   * Manages the current conversation session ID.
   */
  public readonly session: SessionManager;

  private readonly projectManager: ProjectManager;

  /**
   * The underlying Axon instance used for LLM middleware hooks.
   */
  public readonly axon: Axon;

  /**
   * The native Rust engine handling BYODB math and queueing.
   */
  public readonly engine: NativeEngine;

  private readonly api: Api;
  private readonly collectorApi: Api;

  private readonly recallEngine: RecallEngine;
  private readonly persistenceEngine: PersistenceEngine;
  private readonly augmentationEngine: AugmentationEngine;

  /**
   * Access the LLM integration layer.
   */
  public readonly llm = {
    /**
     * Registers a third-party LLM client (e.g., OpenAI, Anthropic) with Memori.
     * This enables Memori to automatically inject recalled memories into the system prompt.
     *
     * @param client - An instantiated client from a supported provider (OpenAI, Anthropic, etc).
     */
    register: (client: unknown): Memori => {
      this.axon.llm.register(client);
      return this;
    },
  };

  /**
   * Access augmentation lifecycle helpers.
   *
   * Mirrors the Python API (`mem.augmentation.wait()`), while delegating to the
   * native engine's queue flush behavior in TypeScript BYODB mode.
   */
  public readonly augmentation = {
    wait: (timeoutMs?: number): Promise<boolean> => this.engine.waitForAugmentation(timeoutMs),
  };

  constructor(options: MemoriOptions = {}) {
    // 1. Core State
    this.config = new Config();
    this.session = new SessionManager();
    this.projectManager = new ProjectManager();
    this.axon = new Axon();

    // 2. Network Layer
    this.api = new Api(this.config, ApiSubdomain.DEFAULT);
    this.collectorApi = new Api(this.config, ApiSubdomain.COLLECTOR);

    // 3. Local Rust Layer & Storage Manager Init
    if (options.conn) {
      this.config.storage = new StorageManager(options.conn);
      this.engine = new NativeEngine(this.config.storage, options.embeddingModel);
      // Give the storage manager access to the engine's fastembed so it can
      // embed facts that arrive without embeddings from the augmentation API.
      this.config.storage.setEmbedder(this.engine.embedTexts.bind(this.engine));
      // Ensure storage.close() tears down native worker runtimes so short-lived
      // scripts can exit without lingering background handles.
      this.config.storage.setEngineShutdown(this.engine.shutdown.bind(this.engine));
    } else {
      this.engine = new NativeEngine(undefined, options.embeddingModel);
    }

    // 4. Engines (Now receiving both the Cloud API and Local Engine)
    this.recallEngine = new RecallEngine(this.api, this.engine, this.config, this.session);
    this.persistenceEngine = new PersistenceEngine(
      this.api,
      this.engine,
      this.config,
      this.session
    );
    this.augmentationEngine = new AugmentationEngine(
      this.collectorApi,
      this.engine,
      this.config,
      this.session,
      this.projectManager
    );

    // 5. Register Hooks
    this.axon.hook.before(this.recallEngine.handleRecall.bind(this.recallEngine));
    this.axon.hook.after(this.persistenceEngine.handlePersistence.bind(this.persistenceEngine));
    this.axon.hook.after(this.augmentationEngine.handleAugmentation.bind(this.augmentationEngine));
  }

  /**
   * Configures the attribution context for subsequent operations.
   * This helps segregate memories by user (Entity) or workflow (Process).
   *
   * @param entityId - Unique identifier for the end-user (e.g., user GUID).
   * @param processId - Unique identifier for the specific workflow or agent.
   */
  public attribution(entityId?: string, processId?: string): this {
    if (entityId) this.config.entityId = entityId;
    if (processId) this.config.processId = processId;
    return this;
  }

  /**
   * Manually retrieves relevant facts from Memori based on a query.
   * Useful if you need to fetch memories without triggering a full LLM completion.
   *
   * @param query - The search text used to find relevant memories.
   * @returns A list of parsed facts with their relevance scores.
   */
  public async recall(query: string): Promise<ParsedFact[]> {
    return this.recallEngine.recall(query);
  }

  /**
   * Resets the current session ID to a new random UUID.
   * Call this when starting a completely new conversation thread.
   */
  public resetSession(): this {
    this.session.reset();
    return this;
  }

  /**
   * Manually sets the session ID.
   * Use this to resume an existing conversation thread from your database.
   *
   * @param id - The UUID of the session to resume.
   */
  public setSession(id: string): this {
    this.session.set(id);
    return this;
  }

  /**
   * Securely attaches a supported framework integration to this Memori instance.
   *
   * @typeParam T - The type of integration being created
   * @param IntegrationClass - The integration class constructor to instantiate
   * @returns A new instance of the specified integration with access to Memori's core engines
   */
  public integrate<T extends SupportedIntegration>(IntegrationClass: IntegrationConstructor<T>): T {
    return new IntegrationClass({
      recall: this.recallEngine,
      persistence: this.persistenceEngine,
      augmentation: this.augmentationEngine,
      config: this.config,
      session: this.session,
      project: this.projectManager,
    });
  }
}
