import { Axon } from '@memorilabs/axon';
import { Config } from './core/config.js';
import { SessionManager } from './core/session.js';
import { Api, ApiSubdomain } from './core/network.js';
import { RecallEngine } from './engines/recall.js';
import { PersistenceEngine } from './engines/persistence.js';
import { AugmentationEngine } from './engines/augmentation.js';
import { ParsedFact } from './types/api.js';

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

  /**
   * The underlying Axon instance used for LLM middleware hooks.
   */
  public readonly axon: Axon;

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
    register: (client: unknown): this => {
      this.axon.llm.register(client);
      return this;
    },
  };

  constructor() {
    // 1. Core State
    this.config = new Config();
    this.session = new SessionManager();
    this.axon = new Axon();

    // 2. Network Layer
    this.api = new Api(this.config, ApiSubdomain.DEFAULT);
    this.collectorApi = new Api(this.config, ApiSubdomain.COLLECTOR);

    // 3. Engines
    this.recallEngine = new RecallEngine(this.api, this.config, this.session);
    this.persistenceEngine = new PersistenceEngine(this.api, this.config, this.session);
    this.augmentationEngine = new AugmentationEngine(this.collectorApi, this.config, this.session);

    // 4. Register Hooks
    this.axon.before.register(this.recallEngine.handleRecall.bind(this.recallEngine));
    this.axon.after.register(this.persistenceEngine.handlePersistence.bind(this.persistenceEngine));
    this.axon.after.register(
      this.augmentationEngine.handleAugmentation.bind(this.augmentationEngine)
    );
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
}
