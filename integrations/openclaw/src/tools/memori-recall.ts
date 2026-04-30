import { createRecallClient } from '../utils/memori-client.js';
import type { ToolDeps } from './types.js';

export function createMemoriRecallTool(deps: ToolDeps) {
  const { config, logger } = deps;

  return {
    name: 'memori_recall',
    label: 'Recall Memory',
    description:
      'CRITICAL: You MUST use this tool to search for past context BEFORE claiming you do not know the user, their preferences, or past events. Explicitly fetch relevant memories from Memori using filters...',
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
          description:
            'ISO 8601 (MUST be UTC) date string to filter memories created on or after this time',
        },
        dateEnd: {
          type: 'string',
          description:
            'ISO 8601 (MUST be UTC) date string to filter memories created on or before this time',
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
          description: 'Filter to a specific fact signal. MUST be one of the allowed enum values.',
          enum: [
            'commit',
            'discovery',
            'failure',
            'inference',
            'pattern',
            'result',
            'update',
            'verification',
          ],
        },
        source: {
          type: 'string',
          description:
            'Filter to a specific source origin. MUST be one of the allowed enum values.',
          enum: [
            'constraint',
            'decision',
            'execution',
            'fact',
            'insight',
            'instruction',
            'status',
            'strategy',
            'task',
          ],
        },
      },
      // Force the LLM to ALWAYS provide a search query
      required: ['query'],
    },

    async execute(
      _toolCallId: string,
      params: {
        query: string;
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
          const errorResult = { error: 'sessionId cannot be provided without projectId' };
          logger.warn(`memori_recall rejected: ${JSON.stringify(errorResult)}`);
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
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
        const errorResult = { error: 'Recall failed' };
        return {
          content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
          details: null,
        };
      }
    },
  };
}
