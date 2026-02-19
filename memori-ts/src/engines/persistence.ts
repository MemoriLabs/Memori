import { CallContext, LLMRequest, LLMResponse, Message } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';

export class PersistenceEngine {
  constructor(
    private readonly api: Api,
    private readonly config: Config,
    private readonly session: SessionManager
  ) {}

  public async handlePersistence(
    req: LLMRequest,
    res: LLMResponse,
    _ctx: CallContext
  ): Promise<LLMResponse> {
    const sessionId = this.session.id;
    if (!sessionId) return res;

    const lastUserMessage = this.extractLastUserMessage(req.messages);
    if (!lastUserMessage) return res;

    const payload = {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      messages: [
        { role: 'user', type: 'text', text: lastUserMessage },
        { role: 'assistant', type: 'text', text: res.content },
      ],
      session: { id: sessionId },
    };

    try {
      await this.api.post('cloud/conversation/messages', payload);
    } catch (e) {
      console.warn('Memori Persistence failed:', e);
    }
    return res;
  }

  private extractLastUserMessage(messages: Message[]): string | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') return messages[i].content;
    }
    return undefined;
  }
}
