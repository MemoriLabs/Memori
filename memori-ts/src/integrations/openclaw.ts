import { BaseIntegration } from './base.js';

/**
 * OpenClaw-specific integration for Memori.
 * Provides memory capture and recall functionality for OpenClaw agents.
 */
export class OpenClawIntegration extends BaseIntegration {
  /**
   * Sets the attribution context for memory operations.
   *
   * @param entityId - Unique identifier for the entity (required)
   * @param processId - Optional identifier for the workflow/process/agent
   * @returns This instance for method chaining
   */
  public setAttribution(entityId: string, processId?: string): this {
    this.core.config.entityId = entityId;
    if (processId) this.core.config.processId = processId;
    return this;
  }

  /**
   * Sets the current conversation session ID.
   *
   * @param sessionId - Unique session identifier
   * @returns This instance for method chaining
   */
  public setSession(sessionId: string): this {
    this.core.session.set(sessionId);
    return this;
  }

  /**
   * Captures a conversation turn and sends it to Memori for processing.
   *
   * @param userMessage - The user's input message
   * @param agentResponse - The agent's complete response
   * @returns Promise that resolves when capture is complete
   *
   * @throws Does not throw - errors are logged but swallowed to prevent disrupting the agent
   */
  public async capture(userMessage: string, agentResponse: string): Promise<void> {
    await this.executeCapture(userMessage, agentResponse);
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
}
