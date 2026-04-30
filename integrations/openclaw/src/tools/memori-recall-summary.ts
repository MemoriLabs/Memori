import { createRecallClient } from '../utils/memori-client.js';
import type { ToolDeps } from './types.js';

export function createMemoriRecallSummaryTool(deps: ToolDeps) {
  const { config, logger } = deps;

  return {
    name: 'memori_recall_summary',
    label: 'Recall Memory Summary',
    description:
      'CRITICAL: You MUST use this tool BEFORE answering any requests for a summary, status update, daily brief, or high-level overview of a project or past sessions. Fetch summarized views of stored memories from Memori within a specific date range.',
    parameters: {
      type: 'object',
      properties: {
        dateStart: {
          type: 'string',
          description:
            'ISO 8601 (MUST be UTC) date string to filter summaries created on or after this time',
        },
        dateEnd: {
          type: 'string',
          description:
            'ISO 8601 (MUST be UTC) date string to filter summaries created on or before this time',
        },
        projectId: {
          type: 'string',
          description:
            'CRITICAL: Leave this EMPTY to use the configured default project. ONLY provide a value if the user explicitly asks to search a different project by name.',
        },
        sessionId: {
          type: 'string',
          description: 'Filter to a specific session. Cannot be used without projectId.',
        },
      },
    },

    async execute(
      _toolCallId: string,
      params: {
        dateStart?: string;
        dateEnd?: string;
        projectId?: string;
        sessionId?: string;
      }
    ) {
      try {
        const finalParams = { projectId: config.projectId, ...params };

        if (finalParams.sessionId && !finalParams.projectId) {
          const errorResult = { error: 'sessionId cannot be provided without projectId' };
          logger.warn(`memori_recall_summary rejected: ${JSON.stringify(errorResult)}`);
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
            details: null,
          };
        }

        logger.info(`memori_recall_summary params: ${JSON.stringify(finalParams)}`);
        const client = createRecallClient(config.apiKey, config.entityId);
        const result = await client.agentRecallSummary(finalParams);

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          details: null,
        };
      } catch (e) {
        logger.warn(`memori_recall_summary failed: ${String(e)}`);
        const errorResult = { error: 'Recall summary failed' };
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
          details: null,
        };
      }
    },
  };
}
