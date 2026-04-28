import { createRecallClient } from '../utils/memori-client.js';
import type { ToolDeps } from './types.js';

export function createMemoriRecallTool(deps: ToolDeps) {
  const { config, logger } = deps;

  return {
    name: 'memori_recall',
    label: 'Recall Memory',
    description:
      'Explicitly fetch relevant memories from Memori using filters like date, project, session, signal, and source.',
    parameters: {
      type: 'object',
      properties: {
        dateStart: {
          type: 'string',
          description: 'ISO 8601 date string to filter memories created on or after this time',
        },
        dateEnd: {
          type: 'string',
          description: 'ISO 8601 date string to filter memories created on or before this time',
        },
        projectId: {
          type: 'string',
          description: 'Filter to a specific project. Defaults to the current project.',
        },
        sessionId: {
          type: 'string',
          description: 'Filter to a specific session. Requires projectId to also be provided.',
        },
        signal: {
          type: 'string',
          description: 'Filter to a specific fact signal (e.g., system, user, derived)',
        },
        source: {
          type: 'string',
          description: 'Filter to a specific source origin',
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
        signal?: string;
        source?: string;
      }
    ) {
      try {
        logger.info(`memori_recall params: ${JSON.stringify(params)}`);
        const client = createRecallClient(config.apiKey, config.entityId);
        const result = await client.agentRecall(params);
        return { content: [{ type: 'text' as const, text: JSON.stringify(result) }], details: null };
      } catch (e) {
        logger.warn(`memori_recall failed: ${String(e)}`);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify({ error: 'Recall failed' }) }],
          details: null,
        };
      }
    },
  };
}
