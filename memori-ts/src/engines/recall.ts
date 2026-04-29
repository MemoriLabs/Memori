import { CallContext, LLMRequest, Message, Role } from '@memorilabs/axon';
import { Api } from '../core/network.js';
import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';
import { ProjectManager } from '../core/project.js';
import { NativeEngine } from '../core/engine.js';
import {
  extractFacts,
  extractHistory,
  extractLastUserMessageString,
  formatSummariesFromFacts,
  stringifyContent,
} from '../utils/utils.js';
import {
  AgentRecallParams,
  AgentRecallResponse,
  AgentRecallSummaryParams,
  AgentRecallSummaryResponse,
  CloudRecallResponse,
  ParsedFact,
} from '../types/api.js';

type RawHistoryMessage = {
  role?: unknown;
  content?: unknown;
  text?: unknown;
};

function sanitizeHistoryMessages(messages: RawHistoryMessage[], dropSystem = false): Message[] {
  const sanitized: Message[] = [];

  for (const message of messages) {
    const roleValue = message.role === 'model' ? 'assistant' : message.role;
    const role = typeof roleValue === 'string' ? roleValue : 'user';
    const content = stringifyContent(message.content ?? message.text);

    if (dropSystem && role === 'system') continue;
    if (role === 'tool') continue;
    if (role === 'assistant' && !content.trim()) continue;

    sanitized.push({ role: role as Role, content });
  }

  return sanitized;
}

/**
 * Retrieves relevant memories and injects them into the LLM system prompt before each call.
 *
 * Operates in two modes: local (BYODB — vector search via the Rust engine) or cloud
 * (API call to Memori's recall endpoint). Also re-hydrates conversation history when
 * available.
 */
export class RecallEngine {
  constructor(
    private readonly api: Api,
    private readonly engine: NativeEngine,
    private readonly config: Config,
    private readonly session: SessionManager,
    private readonly project: ProjectManager
  ) {}

  /**
   * Manually triggers a memory retrieval.
   * Routes to the local Rust engine if storage is active, otherwise hits the Cloud API.
   */
  public async recall(query: string): Promise<ParsedFact[]> {
    if (this.engine.hasStorage) {
      if (!this.config.entityId) return [];
      try {
        return await this.retrieveLocal(query);
      } catch (e) {
        console.warn('Local Manual Recall failed:', e);
        return [];
      }
    }

    try {
      const { facts } = await this.retrieveCloud(query);
      return facts;
    } catch (e) {
      console.warn('Memori Manual Recall failed:', e);
      return [];
    }
  }

  /**
   * Manually fetches memories from the agent recall endpoint (GET /v1/agent/recall).
   * Project ID defaults to the current project context; session ID must be explicitly
   * provided and requires a project ID to be present.
   */
  public async agentRecall(params: AgentRecallParams = {}): Promise<AgentRecallResponse> {
    const projectId = params.projectId ?? this.project.id;
    const sessionId = params.sessionId;

    if (sessionId && !projectId) {
      throw new Error('sessionId cannot be provided without projectId');
    }

    const qs = this.buildQueryString({
      date_start: params.dateStart,
      date_end: params.dateEnd,
      entity_id: this.config.entityId,
      project_id: projectId,
      session_id: sessionId,
      signal: params.signal,
      source: params.source,
    });

    return this.api.get<AgentRecallResponse>(`agent/recall${qs}`);
  }

  /**
   * Fetches memory summaries from the agent recall summary endpoint
   * (GET /v1/agent/recall/summary). Project ID defaults to the current project
   * context; session ID must be explicitly provided and requires a project ID.
   */
  public async agentRecallSummary(
    params: AgentRecallSummaryParams = {}
  ): Promise<AgentRecallSummaryResponse> {
    const projectId = params.projectId ?? this.project.id;
    const sessionId = params.sessionId;

    if (sessionId && !projectId) {
      throw new Error('sessionId cannot be provided without projectId');
    }

    const qs = this.buildQueryString({
      date_start: params.dateStart,
      date_end: params.dateEnd,
      project_id: projectId,
      session_id: sessionId,
    });

    return this.api.get<AgentRecallSummaryResponse>(`agent/recall/summary${qs}`);
  }

  private buildQueryString(
    params: Record<string, string | number | boolean | Date | null | undefined>
  ): string {
    const qs = new URLSearchParams();

    for (const [key, value] of Object.entries(params)) {
      if (value != null && value !== '') {
        if (value instanceof Date) {
          // Properly serialize Date objects to ISO 8601 strings for the backend
          qs.set(key, value.toISOString());
        } else {
          qs.set(key, String(value));
        }
      }
    }

    const str = qs.toString();
    return str ? `?${str}` : '';
  }

  /**
   * The Axon 'before' hook that injects memories into the LLM system prompt.
   */
  public async handleRecall(req: LLMRequest, _ctx: CallContext): Promise<LLMRequest> {
    const sessionId = this.session.id;
    if (!sessionId) return req;

    const userQuery = extractLastUserMessageString(req.messages);
    if (!userQuery) return req;

    let facts: ParsedFact[] = [];
    let historyMessages: Message[] = [];

    if (this.engine.hasStorage) {
      if (!this.config.entityId) return req;
      try {
        // Fetch long-term vector facts from the Rust core
        facts = await this.retrieveLocal(userQuery);

        if (this.config.storage) {
          const rawHistory = await this.config.storage.getConversationHistory(sessionId);

          historyMessages = sanitizeHistoryMessages(rawHistory);
        }
      } catch (e) {
        console.warn('Local Recall Hook failed:', e);
        return req;
      }
    } else {
      try {
        ({ facts, history: historyMessages } = await this.retrieveCloud(userQuery));
      } catch (e) {
        console.warn('Memori Recall failed:', e);
        return req;
      }
    }

    const relevantFacts = facts
      .filter((f) => f.score >= this.config.recallRelevanceThreshold)
      .map((f) => {
        const dateSuffix = f.dateCreated ? `. Stated at ${f.dateCreated}` : '';
        return `- ${f.content}${dateSuffix}`;
      });

    const relevantSummaries = formatSummariesFromFacts(
      facts.filter((f) => f.score >= this.config.recallRelevanceThreshold)
    );

    let messages = [...req.messages];

    // Prepend the short-term conversation history to the prompt
    if (historyMessages.length > 0) {
      messages = [...historyMessages, ...messages];
    }

    // Inject the relevant long-term semantic facts into the system prompt
    if (relevantFacts.length > 0) {
      let contextBody = `Relevant context about the user:\n${relevantFacts.join('\n')}`;
      if (relevantSummaries.length > 0) {
        contextBody += `\n\n## Summaries\n\n${relevantSummaries.join('\n\n')}`;
      }

      const recallContext = `\n\n<memori_context>\nOnly use the relevant context if it is relevant to the user's query. ${contextBody}\n</memori_context>`;

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

  private async retrieveLocal(query: string): Promise<ParsedFact[]> {
    const results = await this.engine.retrieve({
      entity_id: this.config.entityId || '',
      query_text: query,
      dense_limit: 100,
      limit: 10,
    });
    return results.map((r) => ({
      content: r.content,
      score: r.rank_score ?? r.similarity ?? 0,
      dateCreated: r.date_created,
      summaries: r.summaries?.map((s) => ({
        content: s.content,
        dateCreated: s.date_created,
      })),
    }));
  }

  private async retrieveCloud(query: string): Promise<{ facts: ParsedFact[]; history: Message[] }> {
    const payload = {
      attribution: {
        entity: { id: this.config.entityId },
        process: { id: this.config.processId },
      },
      query,
      session: { id: this.session.id },
    };
    const response = await this.api.post<CloudRecallResponse>('cloud/recall', payload);
    const facts = extractFacts(response);
    const history = sanitizeHistoryMessages(extractHistory(response) as RawHistoryMessage[], true);
    return { facts, history };
  }
}
