import { QuotaExceededError } from '@memorilabs/memori';
import { createRecallClient } from '../utils/memori-client.js';
import type { ToolDeps } from './types.js';

export function createMemoriCompactionTool(deps: ToolDeps) {
  const { config, logger } = deps;

  return {
    name: 'memori_compaction',
    label: 'Compact Agent Memory',
    description: `Use this tool to retrieve a full structured snapshot of the agent's long-term memory at the START of a new session, after a context reset, or when resuming work from a prior conversation.

The compaction returns:
- **state**: active_tasks (work in progress), open_loops (unresolved threads), pending_results
- **standing_orders**: persistent instructions the agent must continue to follow
- **environment**: environment variable context captured during prior sessions
- **workspace_changes**: recent file or system changes made by the agent
- **continuation**: last_action (what the agent did last) and next_expected_action (what it should do next)
- **messages**: a tail of recent conversation messages for continuity
- **timeline**: a chronological narrative of agent activity (when available)

WHEN TO USE:
- At the start of a new conversation when the user is resuming prior work
- When the user says "pick up where we left off", "continue the task", "what was I working on?", or similar
- When context has been compacted or reset and you need to reconstruct the full agent state before proceeding

WHEN NOT TO USE:
- Do NOT call this on every turn — it costs 100 memory credits per execution
- Do NOT use this for targeted memory search — use memori_recall for that instead
- Do NOT call this if the user is starting a brand-new task with no prior context`,

    parameters: {
      type: 'object',
      required: ['projectId'],
      properties: {
        projectId: {
          type: 'string',
          description:
            'The project to compact. REQUIRED — always pass the configured project ID. This scopes the compaction to the correct workspace.',
        },
        sessionId: {
          type: 'string',
          description:
            'Scope the compaction to a specific agent session. Leave empty to compact across all sessions in the project. Cannot be used without projectId.',
        },
        numMessages: {
          type: 'number',
          description:
            'Number of recent conversation messages to include in the result. Defaults to 5. Increase (up to ~20) only if the user explicitly asks for more conversation context.',
        },
      },
    },

    async execute(
      _toolCallId: string,
      params: {
        projectId?: string;
        sessionId?: string;
        numMessages?: number;
      }
    ) {
      try {
        // Config projectId is the fallback; an explicit LLM-provided value overrides it.
        const finalParams = { projectId: config.projectId, ...params };

        if (!finalParams.projectId) {
          const errorResult = { error: 'projectId is required but was not configured' };
          logger.warn(`memori_compaction rejected: ${JSON.stringify(errorResult)}`);
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
            details: null,
          };
        }

        if (finalParams.sessionId && !finalParams.projectId) {
          const errorResult = { error: 'sessionId cannot be provided without projectId' };
          logger.warn(`memori_compaction rejected: ${JSON.stringify(errorResult)}`);
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
            details: null,
          };
        }

        logger.info(`memori_compaction params: ${JSON.stringify(finalParams)}`);
        const client = createRecallClient(config.apiKey, config.entityId);
        const result = await client.agentCompaction(finalParams);

        if (result === null) {
          const errorResult = { error: 'Compaction failed' };
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
            details: null,
          };
        }

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          details: null,
        };
      } catch (e) {
        if (e instanceof QuotaExceededError) {
          logger.warn('memori_compaction quota exceeded');
          const errorResult = {
            error:
              'Quota exceeded: compaction costs 100 memory credits and your organization has exhausted its Recall Execution quota.',
          };
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
            details: null,
          };
        }

        logger.warn(`memori_compaction failed: ${String(e)}`);
        const errorResult = { error: 'Compaction failed' };
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
          details: null,
        };
      }
    },
  };
}
