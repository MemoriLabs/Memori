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
        query: {
          type: 'string',
          description:
            'REQUIRED: The natural language search query to find specific facts (e.g., "What database did we decide to use?", "Ryan\'s dogs"). DO NOT use wildcards like "*" or regex. This is a semantic search, so use real words.',
        },
        limit: {
          type: 'number',
          description: 'Maximum number of memories to return (default: 10)',
        },
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
          description:
            'CRITICAL: Leave this EMPTY to use the configured default project. ONLY provide a value if the user explicitly asks to search a different project by name.',
        },
        sessionId: {
          type: 'string',
          description: 'Filter to a specific session. Cannot be used without projectId.',
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
      // Force the LLM to ALWAYS provide a search query
      required: ['query'],
    },

    async execute(
      _toolCallId: string,
      params: {
        query: string;
        limit?: number;
        dateStart?: string;
        dateEnd?: string;
        projectId?: string;
        sessionId?: string;
        signal?: string;
        source?: string;
      }
    ) {
      try {
        // If params.projectId is undefined, it falls back to config.projectId.
        // If the LLM intentionally provides one, it overwrites the config.
        const finalParams = { projectId: config.projectId, ...params };

        if (finalParams.sessionId && !finalParams.projectId) {
          return {
            content: [
              {
                type: 'text' as const,
                text: JSON.stringify({ error: 'sessionId cannot be provided without projectId' }),
              },
            ],
            details: null,
          };
        }

        logger.info(`memori_recall params: ${JSON.stringify(finalParams)}`);
        const client = createRecallClient(config.apiKey, config.entityId);
        const result = await client.agentRecall(finalParams);
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          details: null,
        };
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
