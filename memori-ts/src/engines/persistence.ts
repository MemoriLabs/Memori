import { CallContext, LLMRequest, LLMResponse } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';
import { NativeEngine } from '../core/engine.js';
import { extractLastUserMessage } from '../utils/utils.js';
import { WriteBatch } from 'src/types/storage.js';

/**
 * Saves conversation messages to the Memori Cloud after each LLM response.
 *
 * Skipped entirely when a local storage connection is present — in BYODB mode
 * the Rust augmentation pipeline writes conversation history directly to the database.
 */
export class PersistenceEngine {
  constructor(
    private readonly api: Api,
    private readonly engine: NativeEngine,
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

    const lastUserMessage = extractLastUserMessage(req.messages);
    if (!lastUserMessage) return res;

    if (this.engine.hasStorage && this.config.storage) {
      const batch: WriteBatch = {
        ops: [
          {
            op_type: 'conversation_message.create',
            payload: {
              conversation_id: sessionId,
              messages: [
                { role: 'user', content: lastUserMessage },
                { role: 'assistant', content: res.content },
              ],
            },
          },
        ],
      };

      await this.config.storage.writeBatch(batch);

      return res;
    }

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
}
