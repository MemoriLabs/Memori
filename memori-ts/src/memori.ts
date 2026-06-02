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
import type { ConnFactory } from './storage/base.js';

export interface RequestScopeOptions {
  entityId?: string;
  processId?: string;
  sessionId?: string;
}

/**
 * A lightweight, per-request view over a shared Memori instance.
 *
 * Holds its own `entityId`, `processId`, and session so concurrent requests
 * cannot bleed attribution state into each other. The expensive resources
 * (NativeEngine, ONNX model, worker runtimes, StorageManager) are shared with
 * the parent Memori instance and are never reloaded.
 *
 * Obtain one via `memori.forRequest(options)` rather than constructing directly.
 */
export class MemoriRequestScope {
  public readonly config: Config;
  public readonly session: SessionManager;
  public readonly axon: Axon;

  private readonly recallEngine: RecallEngine;
  private readonly persistenceEngine: PersistenceEngine;
  private readonly augmentationEngine: AugmentationEngine;
  private readonly sharedEngine: NativeEngine;
  private readonly sharedApi: Api;
  private readonly sharedCollectorApi: Api;
  private readonly sharedProjectManager: ProjectManager;

  /** @internal — use `Memori.forRequest()` instead */
  constructor(
    sharedEngine: NativeEngine,
    sharedApi: Api,
    sharedCollectorApi: Api,
    sharedProjectManager: ProjectManager,
    parentConfig: Config,
    options?: RequestScopeOptions
  ) {
    this.sharedEngine = sharedEngine;
    this.sharedApi = sharedApi;
    this.sharedCollectorApi = sharedCollectorApi;
    this.sharedProjectManager = sharedProjectManager;

    // Copy shared/infrastructure fields; override only per-request identity.
    this.config = new Config();
    this.config.apiKey = parentConfig.apiKey;
    this.config.baseUrl = parentConfig.baseUrl;
    this.config.testMode = parentConfig.testMode;
    this.config.recallRelevanceThreshold = parentConfig.recallRelevanceThreshold;
    this.config.timeout = parentConfig.timeout;
    this.config.storage = parentConfig.storage;
    // Inherit parent attribution as defaults so forRequest({ sessionId }) doesn't
    // silently drop entityId/processId set via memori.attribution() on the shared instance.
    if (parentConfig.entityId) this.config.entityId = parentConfig.entityId;
    if (parentConfig.processId) this.config.processId = parentConfig.processId;
    // Per-request options override the inherited defaults.
    if (options?.entityId) this.config.entityId = options.entityId;
    if (options?.processId) this.config.processId = options.processId;

    this.session = new SessionManager();
    if (options?.sessionId) this.session.set(options.sessionId);

    this.axon = new Axon();
    this.recallEngine = new RecallEngine(
      sharedApi,
      sharedEngine,
      this.config,
      this.session,
      sharedProjectManager
    );
    this.persistenceEngine = new PersistenceEngine(
      sharedApi,
      sharedEngine,
      this.config,
      this.session
    );
    this.augmentationEngine = new AugmentationEngine(
      sharedApi,
      sharedCollectorApi,
      sharedEngine,
      this.config,
      this.session,
      sharedProjectManager
    );

    this.axon.hook.before(this.recallEngine.handleRecall.bind(this.recallEngine));
    this.axon.hook.after(this.persistenceEngine.handlePersistence.bind(this.persistenceEngine));
    this.axon.hook.after(this.augmentationEngine.handleAugmentation.bind(this.augmentationEngine));
  }

  public attribution(entityId?: string, processId?: string): this {
    if (entityId) this.config.entityId = entityId;
    if (processId) this.config.processId = processId;
    return this;
  }

  public async recall(query: string): Promise<ParsedFact[]> {
    return this.recallEngine.recall(query);
  }

  public setSession(id: string): this {
    this.session.set(id);
    return this;
  }

  public resetSession(): this {
    this.session.reset();
    return this;
  }

  public readonly llm = {
    register: (client: unknown): MemoriRequestScope => {
      this.axon.llm.register(client);
      return this;
    },
  };

  public readonly augmentation = {
    wait: (timeoutMs?: number): Promise<boolean> =>
      this.sharedEngine.waitForAugmentation(timeoutMs),
  };

  public integrate<T extends SupportedIntegration>(IntegrationClass: IntegrationConstructor<T>): T {
    return new IntegrationClass({
      recall: this.recallEngine,
      persistence: this.persistenceEngine,
      augmentation: this.augmentationEngine,
      config: this.config,
      session: this.session,
      project: this.sharedProjectManager,
      defaultApi: this.sharedApi,
      collectorApi: this.sharedCollectorApi,
    });
  }
}

export interface MemoriOptions {
  conn?: ConnFactory; // A factory function returning your database connection: () => pool, () => db, etc.
  embeddingModel?: string;
  dialect?: 'sqlite' | 'postgresql' | 'cockroachdb' | 'mysql'; // Override auto-detected SQL dialect (required for CockroachDB).
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

  // These are private but exposed to MemoriRequestScope (same module) via forRequest().
  private readonly projectManager: ProjectManager;
  private readonly api: Api;
  private readonly collectorApi: Api;

  /**
   * The underlying Axon instance used for LLM middleware hooks.
   */
  public readonly axon: Axon;

  /**
   * The native Rust engine handling BYODB math and queueing.
   */
  public readonly engine: NativeEngine;

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
    this.config = new Config();
    this.session = new SessionManager();
    this.projectManager = new ProjectManager();
    this.axon = new Axon();

    this.api = new Api(this.config, ApiSubdomain.DEFAULT);
    this.collectorApi = new Api(this.config, ApiSubdomain.COLLECTOR);

    if (options.conn) {
      this.config.storage = new StorageManager(options.conn, options.dialect);
      this.engine = new NativeEngine(this.config.storage, options.embeddingModel);
      this.config.storage.setEngineShutdown(this.engine.shutdown.bind(this.engine));
      this.config.storage.setEngineBuild(this.engine.build.bind(this.engine));
    } else {
      this.engine = new NativeEngine(undefined, options.embeddingModel);
    }

    this.recallEngine = new RecallEngine(
      this.api,
      this.engine,
      this.config,
      this.session,
      this.projectManager
    );
    this.persistenceEngine = new PersistenceEngine(
      this.api,
      this.engine,
      this.config,
      this.session
    );
    this.augmentationEngine = new AugmentationEngine(
      this.api,
      this.collectorApi,
      this.engine,
      this.config,
      this.session,
      this.projectManager
    );

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
      defaultApi: this.api,
      collectorApi: this.collectorApi,
    });
  }

  /**
   * Creates a lightweight, per-request scope that is safe to use in concurrent
   * Node.js backends.
   *
   * The returned `MemoriRequestScope` has its own `entityId`, `processId`, and
   * session state, so concurrent requests cannot bleed attribution into each other.
   * The expensive resources (ONNX model, worker runtimes, StorageManager) are
   * shared with this instance and are never reloaded.
   *
   * @example
   * ```ts
   * // At app startup — once
   * const memori = new Memori({ conn: () => pool });
   * await memori.config.storage!.build();
   *
   * // In each request handler
   * app.post('/chat', async (req, res) => {
   *   const scope = memori.forRequest({
   *     entityId: req.user.id,
   *     sessionId: req.session.id,
   *   });
   *   scope.axon.llm.register(openai);
   *   const response = await scope.axon.chat.completions.create({ ... });
   *   res.json(response);
   * });
   * ```
   */
  public forRequest(options?: RequestScopeOptions): MemoriRequestScope {
    return new MemoriRequestScope(
      this.engine,
      this.api,
      this.collectorApi,
      this.projectManager,
      this.config,
      options
    );
  }
}
