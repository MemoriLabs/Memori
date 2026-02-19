import { CallContext, LLMRequest, Message } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';
import { extractFacts, extractHistory, stringifyContent } from '../utils/utils.js';
import { HostedRecallResponse, ParsedFact } from '../types/api.js';

export class RecallEngine {
  constructor(
    private readonly api: Api,
    private readonly config: Config,
    private readonly session: SessionManager
  ) {}

  public async recall(query: string): Promise<ParsedFact[]> {
    const payload = {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      query,
      session: { id: this.session.id },
    };

    try {
      const response = await this.api.post<HostedRecallResponse>('cloud/recall', payload);
      return extractFacts(response);
    } catch (e) {
      console.warn('Memori Manual Recall failed:', e);
      return [];
    }
  }

  public async handleRecall(req: LLMRequest, _ctx: CallContext): Promise<LLMRequest> {
    const sessionId = this.session.id;
    if (!sessionId) return req;

    const userQuery = this.extractLastUserMessage(req.messages);
    if (!userQuery) return req;

    const payload = {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      query: userQuery,
      session: { id: sessionId },
    };

    let response: HostedRecallResponse;
    try {
      response = await this.api.post<HostedRecallResponse>('cloud/recall', payload);
    } catch (e) {
      console.warn('Memori Recall failed:', e);
      return req;
    }

    const facts = extractFacts(response);
    const historyRaw = extractHistory(response);

    const historyMessages: Message[] = (historyRaw as Message[])
      .filter((m) => m.role !== 'system')
      .map((m) => ({
        role: m.role,
        content: stringifyContent(m.content),
      }));

    const relevantFacts = facts
      // Filter out low-relevance memories to prevent hallucination or context pollution
      .filter((f) => f.score >= this.config.recallRelevanceThreshold)
      .map((f) => {
        const dateSuffix = f.dateCreated ? `. Stated at ${f.dateCreated}` : '';
        return `- ${f.content}${dateSuffix}`;
      });

    let messages = [...req.messages];

    if (historyMessages.length > 0) {
      messages = [...historyMessages, ...messages];
    }

    if (relevantFacts.length > 0) {
      const factList = relevantFacts.join('\n');
      const recallContext = `\n\n<memori_context>\nOnly use the relevant context if it is relevant to the user's query. Relevant context about the user:\n${factList}\n</memori_context>`;

      const systemIdx = messages.findIndex((m) => m.role === 'system');
      if (systemIdx >= 0) {
        messages[systemIdx] = {
          ...messages[systemIdx],
          content: messages[systemIdx].content + recallContext,
        };
      } else {
        messages.unshift({ role: 'system', content: recallContext });
      }
    }

    return { ...req, messages };
  }

  private extractLastUserMessage(messages: Message[]): string | undefined {
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') return messages[i].content;
    }
    return undefined;
  }
}
