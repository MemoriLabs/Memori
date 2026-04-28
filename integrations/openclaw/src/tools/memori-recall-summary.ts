import { createRecallClient } from '../utils/memori-client.js';
import type { ToolDeps } from './types.js';

export function createMemoriRecallSummaryTool(deps: ToolDeps) {
  const { config, logger } = deps;

  return {
    name: 'memori_recall_summary',
    label: 'Recall Memory Summary',
    description:
      'Fetch summarized views of stored memories from Memori. Useful for getting a high-level overview of what is known about a project or session within a specific date range.',
    parameters: {
      type: 'object',
      properties: {
        dateStart: {
          type: 'string',
          description: 'ISO 8601 date string to filter summaries created on or after this time',
        },
        dateEnd: {
          type: 'string',
          description: 'ISO 8601 date string to filter summaries created on or before this time',
        },
        projectId: {
          type: 'string',
          description: 'Filter to a specific project. Defaults to the current project.',
        },
        sessionId: {
          type: 'string',
          description: 'Filter to a specific session. Requires projectId to also be provided.',
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
        logger.info(`memori_recall_summary params: ${JSON.stringify(params)}`);
        const client = createRecallClient(config.apiKey, config.entityId);
        const result = await client.agentRecallSummary(params);
        return { content: [{ type: 'text' as const, text: JSON.stringify(result) }], details: null };
      } catch (e) {
        logger.warn(`memori_recall_summary failed: ${String(e)}`);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify({ error: 'Recall summary failed' }) }],
          details: null,
        };
      }
    },
  };
}
