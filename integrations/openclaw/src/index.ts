import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { handleRecall } from './handlers/recall.js';
import { handleAugmentation } from './handlers/augmentation.js';
import { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from './types.js';
import { PLUGIN_CONFIG } from './constants.js';
import { MemoriLogger } from './utils/index.js';

const memoriPlugin = {
  id: PLUGIN_CONFIG.ID,
  name: PLUGIN_CONFIG.NAME,
  description: 'Hosted memory backend',

  register(api: OpenClawPluginApi) {
    const rawConfig = api.pluginConfig;

    const config: MemoriPluginConfig = {
      apiKey: rawConfig?.apiKey as string,
      entityId: rawConfig?.entityId as string,
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

    // Register recall hook (before_prompt_build)
    api.on('before_prompt_build', (event: unknown, ctx: unknown) =>
      handleRecall(event as OpenClawEvent, ctx as OpenClawContext, config, logger)
    );

    // Register capture hook (agent_end)
    api.on('agent_end', (event: unknown, ctx: unknown) =>
      handleAugmentation(event as OpenClawEvent, ctx as OpenClawContext, config, logger)
    );
  },
};

export default memoriPlugin;
