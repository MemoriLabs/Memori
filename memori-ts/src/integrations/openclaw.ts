import { IntegrationRequest } from 'src/types/integrations.js';
import {
  AgentRecallParams,
  AgentRecallResponse,
  AgentRecallSummaryParams,
  AgentRecallSummaryResponse,
} from '../types/api.js';
import { BaseIntegration } from './base.js';

/**
 * OpenClaw-specific integration for Memori.
 * Provides memory capture and recall functionality for OpenClaw agents.
 */
export class OpenClawIntegration extends BaseIntegration {
  /**
   * Sets the conversation scope for memory operations.
   *
   * @param sessionId - Unique session identifier
   * @param projectId - Unique identifier for the project
   * @returns This instance for method chaining
   */
  public scope(sessionId: string, projectId: string): this {
    this.core.session.set(sessionId);
    this.core.project.set(projectId);
    return this;
  }

  /**
   * Sets the attribution context for memory operations.
   *
   * @param entityId - Unique identifier for the entity (required)
   * @param processId - Optional identifier for the workflow/process/agent
   * @returns This instance for method chaining
   */
  public attribution(entityId: string, processId?: string): this {
    this.core.config.entityId = entityId;
    if (processId) this.core.config.processId = processId;
    return this;
  }

  /**
   * Captures a conversation turn and sends it to Memori for processing.
   *
   * @param req - The unified integration message containing user text, agent text, and metadata
   * @returns Promise that resolves when capture is complete
   *
   * @throws Does not throw - errors are logged but swallowed to prevent disrupting the agent
   */
  public async augmentation(req: IntegrationRequest): Promise<void> {
    await this.executeAgentAugmentation(req);
  }

  /**
   * Retrieves relevant memories for the given prompt and returns formatted context.
   *
   * @param promptText - The user's prompt/query
   * @returns Formatted XML context string to inject, or undefined if no relevant memories found
   *
   * @throws Does not throw - errors are logged but swallowed, returns undefined on failure
   */
  public async recall(promptText: string): Promise<string | undefined> {
    return this.executeRecall(promptText);
  }

  /**
   * Manually fetches memories from the agent recall endpoint.
   * Project ID defaults to the current project context. Session ID must be explicitly
   * provided and cannot be used without a project ID.
   *
   * @param params - Optional filters: dateStart, dateEnd, projectId, sessionId, signal, source
   * @returns Raw recall response, or empty object on failure
   *
   * @throws Does not throw - errors are logged but swallowed to prevent disrupting the agent
   */
  public async agentRecall(params?: AgentRecallParams): Promise<AgentRecallResponse> {
    return this.executeAgentRecall(params);
  }

  /**
   * Fetches memory summaries from the agent recall summary endpoint.
   * Project ID defaults to the current project context. Session ID must be explicitly
   * provided and cannot be used without a project ID.
   *
   * @param params - Optional filters: dateStart, dateEnd, projectId, sessionId
   * @returns Raw recall summary response, or empty object on failure
   *
   * @throws Does not throw - errors are logged but swallowed to prevent disrupting the agent
   */
  public async agentRecallSummary(
    params?: AgentRecallSummaryParams
  ): Promise<AgentRecallSummaryResponse> {
    return this.executeAgentRecallSummary(params);
  }

  /**
   * Sends feedback directly to the Memori team.
   *
   * @param content - The feedback text
   * @returns Promise that resolves when feedback is sent
   */
  public async agentFeedback(content: string): Promise<void> {
    await this.executeAgentFeedback(content);
  }
}
