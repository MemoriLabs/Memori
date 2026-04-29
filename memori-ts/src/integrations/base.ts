import { LLMRequest, LLMResponse, CallContext } from '@memorilabs/axon';
import { MemoriCore, IntegrationRequest } from '../types/integrations.js';
import {
  AgentRecallParams,
  AgentRecallResponse,
  AgentRecallSummaryParams,
  AgentRecallSummaryResponse,
} from '../types/api.js';

/**
 * Abstract base class for Memori framework integrations.
 *
 * Provides common functionality for translating framework-specific data formats
 * (like OpenClaw messages) into Axon's internal LLM request/response format.
 *
 * @internal
 */
export abstract class BaseIntegration {
  constructor(protected readonly core: MemoriCore) {}

  /**
   * Helper to construct Axon-compatible request payloads from the unified integration format.
   */
  private buildSyntheticPayload(req: IntegrationRequest) {
    const syntheticReq: LLMRequest = {
      messages: [
        {
          role: req.userMessage.role,
          content: req.userMessage.content,
          type: req.userMessage.type,
        },
      ],
      model: req.metadata?.model || '',
    };

    const syntheticRes: LLMResponse = {
      content: req.agentResponse.content,
      type: req.agentResponse.type,
    };

    const syntheticCtx: CallContext = {
      traceId: `integration-trace-${Date.now()}`,
      startedAt: new Date(),
      metadata: req.metadata as unknown as Record<string, unknown>,
    };

    return { syntheticReq, syntheticRes, syntheticCtx };
  }

  /**
   * Internal helper: Captures a conversation turn by translating it into Axon format
   * and feeding it to both the Persistence and Augmentation engines.
   *
   * @param req - The unified integration message containing user text, agent text, and metadata
   * @internal
   */
  protected async executeAugmentation(req: IntegrationRequest): Promise<void> {
    if (!this.core.session.id) return;

    const { syntheticReq, syntheticRes, syntheticCtx } = this.buildSyntheticPayload(req);

    try {
      await this.core.persistence.handlePersistence(syntheticReq, syntheticRes, syntheticCtx);
      await this.core.augmentation.handleAugmentation(syntheticReq, syntheticRes, syntheticCtx);
    } catch (e) {
      console.warn('Memori Integration Capture failed:', e);
    }
  }

  /**
   * Internal helper: Captures an agentic conversation turn by translating it into Axon format
   * and feeding it to both the Persistence and Augmentation engines, including tool traces.
   *
   * @param req - The unified integration message containing user text, agent text, trace, and metadata
   * @internal
   */
  protected async executeAgentAugmentation(req: IntegrationRequest): Promise<void> {
    if (!this.core.session.id) return;

    const { syntheticReq, syntheticRes, syntheticCtx } = this.buildSyntheticPayload(req);

    try {
      await this.core.persistence.handlePersistence(syntheticReq, syntheticRes, syntheticCtx);
      await this.core.augmentation.handleAgentAugmentation(
        syntheticReq,
        syntheticRes,
        syntheticCtx,
        req.trace,
        req.summary
      );
    } catch (e) {
      console.warn('Memori Integration Capture failed:', e);
    }
  }

  /**
   * Internal helper: Fetches memories from the agent recall endpoint.
   *
   * @param params - Optional filter parameters (projectId, sessionId, query, limit)
   * @returns Raw recall response, or empty object on failure
   * @internal
   */
  protected async executeAgentRecall(params?: AgentRecallParams): Promise<AgentRecallResponse> {
    try {
      return await this.core.recall.agentRecall(params);
    } catch (e) {
      console.warn('Memori Agent Recall failed:', e);
      return {};
    }
  }

  /**
   * Internal helper: Fetches memory summaries from the agent recall summary endpoint.
   *
   * @param params - Optional filter parameters (projectId, sessionId, limit)
   * @returns Raw recall summary response, or empty object on failure
   * @internal
   */
  protected async executeAgentRecallSummary(
    params?: AgentRecallSummaryParams
  ): Promise<AgentRecallSummaryResponse> {
    try {
      return await this.core.recall.agentRecallSummary(params);
    } catch (e) {
      console.warn('Memori Agent Recall Summary failed:', e);
      return {};
    }
  }

  /**
   * Internal helper: Recalls memories by translating the query into Axon format,
   * passing it through the Recall engine, and extracting the injected system prompt.
   *
   * @param userMessage - Raw user query text
   * @returns XML-formatted memory context, or undefined if no session or recall fails
   * @internal
   */
  protected async executeRecall(userMessage: string): Promise<string | undefined> {
    if (!this.core.session.id) return undefined;

    const syntheticReq: LLMRequest = {
      messages: [{ role: 'user', content: userMessage }],
    };

    const syntheticCtx: CallContext = {
      traceId: `integration-trace-${Date.now()}`,
      startedAt: new Date(),
      metadata: {},
    };

    try {
      const updatedReq = await this.core.recall.handleRecall(syntheticReq, syntheticCtx);
      const systemMsg = updatedReq.messages.find((m) => m.role === 'system');
      return systemMsg?.content;
    } catch (e) {
      console.warn('Memori Integration Recall failed:', e);
      return undefined;
    }
  }

  /**
   * Internal helper: Sends feedback directly to the Memori team.
   *
   * @param content - The feedback text
   * @internal
   */
  protected async executeAgentFeedback(content: string): Promise<void> {
    try {
      await this.core.defaultApi.post('agent/feedback', { content });
      return;
    } catch (e) {
      console.warn('Memori Agent Feedback failed:', e);
    }
  }
}
