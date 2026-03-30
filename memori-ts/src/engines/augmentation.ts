import { CallContext, LLMRequest, LLMResponse } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';
import { extractLastUserMessage } from '../utils/utils.js';
import { SDK_VERSION } from '../version.js';
import { Trace } from '../types/integrations.js';

export class AugmentationEngine {
  constructor(
    private readonly api: Api,
    private readonly config: Config,
    private readonly session: SessionManager
  ) {}

  private prepareAugmentationData(req: LLMRequest, res: LLMResponse, ctx: CallContext) {
    const sessionId = this.session.id;
    if (!sessionId) return null;

    const lastUserMessage = extractLastUserMessage(req.messages);
    if (!lastUserMessage) return null;

    const messages = [
      { role: 'user', content: lastUserMessage },
      { role: 'assistant', content: res.content },
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

    const payload = {
      conversation: { messages: data.messages, summary: null },
      meta: data.meta,
      session: { id: data.sessionId },
    };

    // Fire-and-forget
    this.api.post('cloud/augmentation', payload).catch((e: unknown) => {
      if (this.config.testMode) console.warn('Augmentation failed:', e);
    });

    return Promise.resolve(res);
  }

  public handleAgentAugmentation(
    req: LLMRequest,
    res: LLMResponse,
    ctx: CallContext,
    trace?: Trace | null,
    summary?: string | null
  ): Promise<LLMResponse> {
    const data = this.prepareAugmentationData(req, res, ctx);
    if (!data) return Promise.resolve(res);

    const payload = {
      conversation: data.messages,
      summary: summary || null,
      trace: trace || null,
      meta: data.meta,
      session: { id: data.sessionId },
    };

    // Fire-and-forget to the dedicated agent endpoint
    this.api.post('agent/augmentation', payload).catch((e: unknown) => {
      if (this.config.testMode) console.warn('Agent Augmentation failed:', e);
    });

    return Promise.resolve(res);
  }

  private buildMeta(req: LLMRequest, ctx: CallContext): Record<string, unknown> {
    return {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      sdk: { lang: 'javascript', version: ctx.metadata.integrationSdkVersion || SDK_VERSION },
      framework: null,
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
      storage: null,
    };
  }
}
