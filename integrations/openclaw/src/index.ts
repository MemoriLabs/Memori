import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { handleRecall } from './handlers/recall.js';
import { handleAugmentation } from './handlers/augmentation.js';
import { OpenClawEvent, OpenClawContext } from './types.js';
import { PLUGIN_CONFIG } from './constants.js';

const memoriPlugin = {
  id: PLUGIN_CONFIG.ID,
  name: PLUGIN_CONFIG.NAME,
  description: 'Hosted memory backend',
  kind: 'memory' as const,

  register(api: OpenClawPluginApi) {
    const config = api.pluginConfig;
    const apiKey = config?.apiKey as string;
    const configuredEntityId = config?.entityId as string;

    if (!apiKey) {
      api.logger.warn(`${PLUGIN_CONFIG.LOG_PREFIX} MEMORI_API_KEY is missing. Plugin disabled.`);

      return;
    }

    api.logger.info(`\n=== ${PLUGIN_CONFIG.LOG_PREFIX} INITIALIZING PLUGIN ===`);
    api.logger.info(
      `${PLUGIN_CONFIG.LOG_PREFIX} Tracking Entity ID: ${configuredEntityId || 'Dynamic (Event-based)'}`
    );

    // Register recall hook (before_prompt_build)
    api.on('before_prompt_build', (event: unknown, ctx: unknown) =>
      handleRecall(event as OpenClawEvent, ctx as OpenClawContext, api, apiKey, configuredEntityId)
    );

    // Register capture hook (agent_end)
    api.on('agent_end', (event: unknown, ctx: unknown) =>
      handleAugmentation(event as OpenClawEvent, ctx as OpenClawContext, api, apiKey, configuredEntityId)
    );
  },
};

export default memoriPlugin;
