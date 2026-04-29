import { createRecallClient } from '../utils/memori-client.js';
import type { ToolDeps } from './types.js';

export function createMemoriFeedbackTool(deps: ToolDeps) {
  const { config, logger } = deps;

  return {
    name: 'memori_feedback',
    label: 'Memori Feedback',
    description:
      'Send feedback directly to the Memori team. Please use this to send us positive and negative feedback - things to fix, feature requests, or just to tell us you love us!',
    parameters: {
      type: 'object',
      properties: {
        content: {
          type: 'string',
          description: 'REQUIRED: The feedback message to send.',
        },
      },
      required: ['content'],
    },

    async execute(
      _toolCallId: string,
      params: {
        content: string;
      }
    ) {
      try {
        logger.info(`memori_feedback sending: ${params.content}`);

        const client = createRecallClient(config.apiKey, config.entityId);
        await client.agentFeedback(params.content);

        const result = { success: true, message: 'Feedback sent successfully.' };

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          details: null,
        };
      } catch (e) {
        logger.warn(`memori_feedback failed: ${String(e)}`);
        const errorResult = { error: 'Feedback failed to send.' };

        logger.info(`memori_feedback error result: ${JSON.stringify(errorResult)}`);

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
          details: null,
        };
      }
    },
  };
}
