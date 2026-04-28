import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { handleRecall } from './handlers/recall.js';
import { handleAugmentation } from './handlers/augmentation.js';
import { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from './types.js';
import { PLUGIN_CONFIG } from './constants.js';
import { MemoriLogger, createRecallClient } from './utils/index.js';

const memoriPlugin = {
  id: PLUGIN_CONFIG.ID,
  name: PLUGIN_CONFIG.NAME,
  description: 'Hosted memory backend',

  register(api: OpenClawPluginApi) {
    const rawConfig = api.pluginConfig;

    const config: MemoriPluginConfig = {
      apiKey: rawConfig?.apiKey as string,
      entityId: rawConfig?.entityId as string,
      projectId: rawConfig?.projectId as string,
    };

    if (!config.apiKey || !config.entityId) {
      api.logger.warn(
        `${PLUGIN_CONFIG.LOG_PREFIX} Missing apiKey or entityId in config. Plugin disabled.`
      );
      return;
    }

    const logger = new MemoriLogger(api);

    logger.info(`\n=== ${PLUGIN_CONFIG.LOG_PREFIX} INITIALIZING PLUGIN ===`);
    logger.info(`${PLUGIN_CONFIG.LOG_PREFIX} Tracking Entity ID: ${config.entityId}`);

    api.on('before_prompt_build', (event: unknown, ctx: unknown) =>
      handleRecall(event as OpenClawEvent, ctx as OpenClawContext, config, logger)
    );

    api.on('agent_end', (event: unknown, ctx: unknown) =>
      handleAugmentation(event as OpenClawEvent, ctx as OpenClawContext, config, logger)
    );

    api.registerTool({
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
      execute: async (
        _toolCallId: string,
        params: {
          dateStart?: string;
          dateEnd?: string;
          projectId?: string;
          sessionId?: string;
          signal?: string;
          source?: string;
        }
      ) => {
        try {
          console.log('memori_recall params', JSON.stringify(params, null, 2));
          const client = createRecallClient(config.apiKey, config.entityId);
          const result = await client.agentRecall(params);
          return { content: [{ type: 'text', text: JSON.stringify(result) }], details: null };
        } catch (e) {
          logger.warn(`memori_recall failed: ${String(e)}`);
          return {
            content: [{ type: 'text', text: JSON.stringify({ error: 'Recall failed' }) }],
            details: null,
          };
        }
      },
    });

    api.registerTool({
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
      execute: async (
        _toolCallId: string,
        params: {
          dateStart?: string;
          dateEnd?: string;
          projectId?: string;
          sessionId?: string;
        }
      ) => {
        try {
          console.log('memori_recall_summary params', JSON.stringify(params, null, 2));
          const client = createRecallClient(config.apiKey, config.entityId);
          const result = await client.agentRecallSummary(params);
          return { content: [{ type: 'text', text: JSON.stringify(result) }], details: null };
        } catch (e) {
          logger.warn(`memori_recall_summary failed: ${String(e)}`);
          return {
            content: [{ type: 'text', text: JSON.stringify({ error: 'Recall summary failed' }) }],
            details: null,
          };
        }
      },
    });
  },
};

export default memoriPlugin;
