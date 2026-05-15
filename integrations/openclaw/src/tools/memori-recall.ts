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
          description:
            'Filter by how the memory was derived. MUST be set together with `source` using one of the allowed (source, signal) pairs — never set independently. Valid pairs: (constraint, discovery), (decision, commit), (fact, verification), (execution, failure), (instruction, discovery), (insight, inference), (status, update), (strategy, pattern), (task, result).',
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
            'Filter by memory type. MUST be set together with `signal` using one of the allowed (source, signal) pairs — never set independently. Valid pairs: (constraint, discovery), (decision, commit), (fact, verification), (execution, failure), (instruction, discovery), (insight, inference), (status, update), (strategy, pattern), (task, result).',
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

        const hasSource = finalParams.source != null;
        const hasSignal = finalParams.signal != null;
        if (hasSource !== hasSignal) {
          const errorResult = {
            error: 'source and signal must be provided together or both omitted',
          };
          logger.warn(`memori_recall rejected: ${JSON.stringify(errorResult)}`);
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
            details: null,
          };
        }

        const VALID_PAIRS: Record<string, string> = {
          constraint: 'discovery',
          decision: 'commit',
          fact: 'verification',
          execution: 'failure',
          instruction: 'discovery',
          insight: 'inference',
          status: 'update',
          strategy: 'pattern',
          task: 'result',
        };
        const source = finalParams.source;
        if (hasSource && source != null && VALID_PAIRS[source] !== finalParams.signal) {
          const errorResult = {
            error: `Invalid (source, signal) pair: (${source}, ${finalParams.signal}). Expected signal for source "${source}" is "${VALID_PAIRS[source]}".`,
          };
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
