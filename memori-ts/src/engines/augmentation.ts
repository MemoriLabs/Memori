import { CallContext, LLMRequest, LLMResponse } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';
import { extractLastUserMessage } from '../utils/utils.js';
import { SDK_VERSION } from '../version.js';

export class AugmentationEngine {
  constructor(
    private readonly api: Api,
    private readonly config: Config,
    private readonly session: SessionManager
  ) {}

  public handleAugmentation(
    req: LLMRequest,
    res: LLMResponse,
    ctx: CallContext
  ): Promise<LLMResponse> {
    const sessionId = this.session.id;
    if (!sessionId) return Promise.resolve(res);

    const lastUserMessage = extractLastUserMessage(req.messages);
    if (!lastUserMessage) return Promise.resolve(res);

    const messages = [
      { role: 'user', content: lastUserMessage },
      { role: 'assistant', content: res.content },
    ];

    const payload = {
      conversation: { messages, summary: null },
      meta: this.buildMeta(req, ctx),
      session: { id: sessionId },
    };

    // Fire-and-forget
    this.api.post('cloud/augmentation', payload).catch((e: unknown) => {
      if (this.config.testMode) console.warn('Augmentation failed:', e);
    });

    return Promise.resolve(res);
  }

  private buildMeta(req: LLMRequest, ctx: CallContext): Record<string, unknown> {
    return {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      sdk: { lang: 'javascript', version: SDK_VERSION },
      framework: null,
      llm: {
        model: {
          provider: ctx.metadata.provider || null,
          sdk: {
            version: ctx.metadata.sdkVersion || null,
          },
        },
      },
      platform: null,
      storage: null,
    };
  }
}
