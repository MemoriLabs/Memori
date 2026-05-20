import { QuotaExceededError } from '@memorilabs/memori';
import { createRecallClient } from '../utils/memori-client.js';
import type { ToolDeps } from './types.js';

export function createMemoriCompactionTool(deps: ToolDeps) {
  const { config, logger } = deps;

  return {
    name: 'memori_compaction',
    label: 'Compact Agent Memory',
    description: `Use this tool to restore working state after context compaction. Returns a structured snapshot of the agent's long-term memory to enable continuation without replaying the full prior session.

WHEN TO USE:
- The agent resumes after compaction
- A long-running workflow has lost conversational detail
- The agent needs to continue operational work without replaying the full prior session
- The agent needs durable state, standing instructions, environment details, open loops, or the next expected action

WHEN NOT TO USE:
- Do NOT call this on every turn — it costs 100 memory credits per execution
- Do NOT use this for targeted memory search — use memori_recall for that instead
- Do NOT call this if the agent is starting a brand-new task with no prior context
- Compaction is not a replacement for precise memory retrieval

The compaction result returns:
- **environment**: environment variable context captured during prior sessions
- **standing_orders**: persistent instructions the agent must continue to follow
- **state**: active_tasks (work in progress), open_loops (unresolved threads), pending_results
- **timeline**: a chronological narrative of agent activity (when available)
- **workspace_changes**: recent file or system changes made by the agent
- **continuation**: last_action (what the agent did last) and next_expected_action (what it should do next)
- **messages**: a tail of recent conversation messages for continuity

HOW TO USE THE RESULT:
Treat the compaction result as the agent's resume state — use it to understand what environment the agent was operating in, which standing orders must continue to be followed, which tasks are active, what happened across the prior session window, and what the agent should do next.

The compaction result should guide continuation, not override explicit user instructions. Before acting on operational details, verify any state that may have changed since compaction.

Pay special attention to:
- Standing orders
- Hard constraints
- Alerting rules
- Expected response formats
- Open loops
- Staleness warnings
- Next expected action

If the compaction result contains a required output format, follow it exactly unless the user gives a newer instruction.`,

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
