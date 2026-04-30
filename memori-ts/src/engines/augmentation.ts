import { CallContext, LLMRequest, LLMResponse } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';
import { ProjectManager } from '../core/project.js';
import { extractLastUserMessageObject } from '../utils/utils.js';
import { SDK_VERSION } from '../version.js';
import { AugmentationInput, Trace } from '../types/integrations.js';
import { NativeEngine } from '../core/engine.js';

type AugmentationData = {
  sessionId: string;
  messages: { role: string; content: string }[];
  meta: Record<string, unknown>;
};

/**
 * Handles sending conversation turns to the augmentation pipeline after each LLM response.
 *
 * Routes to the local Rust engine when a storage connection is present (BYODB mode),
 * otherwise fires a request to the Memori Cloud API.
 */
export class AugmentationEngine {
  constructor(
    private readonly defaultApi: Api,
    private readonly collectorApi: Api,
    private readonly engine: NativeEngine,
    private readonly config: Config,
    private readonly session: SessionManager,
    private readonly project: ProjectManager
  ) {}

  private prepareAugmentationData(req: LLMRequest, res: LLMResponse, ctx: CallContext) {
    const sessionId = this.session.id;
    if (!sessionId) return null;

    const lastUserMessage = extractLastUserMessageObject(req.messages);
    if (!lastUserMessage) return null;

    const messages = [
      { role: 'user', content: lastUserMessage.content, type: lastUserMessage.type ?? 'text' },
      { role: 'assistant', content: res.content, type: res.type ?? 'text' },
    ];

    return {
      sessionId,
      messages,
      meta: this.buildMeta(req, ctx),
    };
  }

  public handleAugmentation(
    req: LLMRequest,
    res: LLMResponse,
    ctx: CallContext
  ): Promise<LLMResponse> {
    const data = this.prepareAugmentationData(req, res, ctx);
    if (!data) return Promise.resolve(res);

    // Route to Rust engine for BYODB processing
    if (this.engine.hasStorage) {
      try {
        this.engine.submitAugmentation(this.buildAugmentationInput(req, ctx, data));
      } catch (e: unknown) {
        if (this.config.testMode) console.warn('Local Augmentation failed:', e);
      }
      return Promise.resolve(res);
    }

    const payload = {
      conversation: { messages: data.messages, summary: null },
      meta: data.meta,
      session: { id: data.sessionId },
    };

    // Fire-and-forget to cloud
    this.collectorApi.post('cloud/augmentation', payload).catch((e: unknown) => {
      if (this.config.testMode) console.warn('Augmentation failed:', e);
    });

    return Promise.resolve(res);
  }

  public async handleAgentAugmentation(
    req: LLMRequest,
    res: LLMResponse,
    ctx: CallContext,
    trace?: Trace | null,
    summary?: string | null
  ): Promise<LLMResponse> {
    const data = this.prepareAugmentationData(req, res, ctx);
    if (!data) return res;

    if (this.engine.hasStorage) {
      try {
        this.engine.submitAugmentation(this.buildAugmentationInput(req, ctx, data));
      } catch (e: unknown) {
        if (this.config.testMode) console.warn('Local Agent Augmentation failed:', e);
      }
      return res;
    }

    const { attribution, ...metaFields } = data.meta;
    const [userMsg, assistantMsg] = data.messages;

    const formattedMessages = [
      {
        role: userMsg.role,
        content: userMsg.content,
        type: userMsg.type,
        trace: null,
      },
      {
        role: assistantMsg.role,
        content: assistantMsg.content,
        type: assistantMsg.type,
        trace: trace ?? null,
      },
    ];

    const turnPayload = {
      attribution,
      messages: formattedMessages,
      project: { id: this.project.id },
      session: { id: data.sessionId },
    };

    const agentAugPayload = {
      attribution,
      conversation: { messages: formattedMessages },
      meta: metaFields,
      project: { id: this.project.id },
      session: { id: data.sessionId, summary: summary ?? null },
      trace: trace ?? null,
    };

    try {
      await this.defaultApi.post('agent/conversation/turn', turnPayload);
    } catch (e: unknown) {
      if (this.config.testMode) console.warn('Agent Conversation Turn failed:', e);
      return res; // Stop execution if the DB write fails!
    }

    this.collectorApi.post('agent/augmentation', agentAugPayload).catch((e: unknown) => {
      if (this.config.testMode) console.warn('Agent Augmentation failed:', e);
    });

    return res;
  }

  private buildMeta(req: LLMRequest, ctx: CallContext): Record<string, unknown> {
    return {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      sdk: { lang: 'javascript', version: ctx.metadata.integrationSdkVersion || SDK_VERSION },
      framework: { provider: null },
      llm: {
        model: {
          provider: ctx.metadata.provider || null,
          sdk: {
            version: ctx.metadata.sdkVersion || null,
          },
          version: req.model || null,
        },
      },
      platform: {
        provider: ctx.metadata.platform || null,
      },
      storage: {
        cockroachdb: false,
        dialect: this.config.storage ? this.config.storage.getDialect() : null,
      },
    };
  }

  private buildAugmentationInput(
    req: LLMRequest,
    ctx: CallContext,
    data: AugmentationData
  ): AugmentationInput {
    return {
      entity_id: this.config.entityId || '',
      process_id: this.config.processId,
      conversation_id: data.sessionId,
      conversation_messages: data.messages,
      llm_provider: ctx.metadata.provider,
      llm_model: req.model,
      llm_provider_sdk_version: ctx.metadata.sdkVersion,
      platform_provider: ctx.metadata.platform,
      sdk_version: (ctx.metadata.integrationSdkVersion as string | undefined) ?? SDK_VERSION,
      session_id: data.sessionId,
      storage_dialect: this.config.storage ? this.config.storage.getDialect() : null,
      storage_cockroachdb: this.config.storage?.getDialect() === 'cockroachdb',
    };
  }
}
