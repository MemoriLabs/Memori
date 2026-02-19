import { CallContext, LLMRequest, LLMResponse, Message } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';

export class AugmentationEngine {
  constructor(
    private readonly api: Api,
    private readonly config: Config,
    private readonly session: SessionManager
  ) {}

  public handleAugmentation(
    req: LLMRequest,
    res: LLMResponse,
    _ctx: CallContext
  ): Promise<LLMResponse> {
    const sessionId = this.session.id;
    if (!sessionId) return Promise.resolve(res);

    const lastUserMessage = this.extractLastUserMessage(req.messages);
    if (!lastUserMessage) return Promise.resolve(res);

    const messages = [
      { role: 'user', content: lastUserMessage },
      { role: 'assistant', content: res.content },
    ];

    const payload = {
      conversation: { messages, summary: null },
      meta: this.buildMeta(),
      session: { id: sessionId },
    };

    // Fire-and-forget
    this.api.post('cloud/augmentation', payload).catch((e: unknown) => {
      if (this.config.testMode) console.warn('Augmentation failed:', e);
    });

    return Promise.resolve(res);
  }

  private extractLastUserMessage(messages: Message[]): string | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') return messages[i].content;
    }
    return undefined;
  }

  private buildMeta(): Record<string, unknown> {
    return {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      sdk: { lang: 'javascript', version: '0.0.1' },
      framework: null,
      llm: null,
      platform: null,
      storage: null,
    };
  }
}
