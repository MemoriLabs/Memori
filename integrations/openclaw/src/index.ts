import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { handleAugmentation } from './handlers/augmentation.js';
import { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from './types.js';
import { PLUGIN_CONFIG } from './constants.js';
import { MemoriLogger, loadSkillsContent } from './utils/index.js';
import { registerAllTools } from './tools/index.js';
import { registerCliCommands } from './cli/commands.js';

const memoriPlugin = {
  id: PLUGIN_CONFIG.ID,
  name: PLUGIN_CONFIG.NAME,
  description: 'Hosted memory backend',

  register(api: OpenClawPluginApi) {
    // 1. Always register CLI commands
    registerCliCommands(api);

    const rawConfig = api.pluginConfig;

    const config: MemoriPluginConfig = {
      apiKey: rawConfig?.apiKey as string,
      entityId: rawConfig?.entityId as string,
      projectId: rawConfig?.projectId as string,
    };

    // 2. Validate configuration
    if (!config.apiKey || !config.entityId) {
      api.logger.warn(
        `${PLUGIN_CONFIG.LOG_PREFIX} Missing apiKey or entityId in config. Plugin disabled.`
      );
      return;
    }

    const logger = new MemoriLogger(api);
    const skillsContent = loadSkillsContent(api.resolvePath.bind(api));

    logger.info(`\n=== ${PLUGIN_CONFIG.LOG_PREFIX} INITIALIZING PLUGIN ===`);
    logger.info(`${PLUGIN_CONFIG.LOG_PREFIX} Tracking Entity ID: ${config.entityId}`);

    // Inject tool usage instructions into the system prompt
    if (skillsContent) {
      api.on('before_prompt_build', () => ({ appendSystemContext: skillsContent }));
    }

    // Augmentation remains automatic
    api.on('agent_end', (event: unknown, ctx: unknown) =>
      handleAugmentation(event as OpenClawEvent, ctx as OpenClawContext, config, logger)
    );

    // Register explicit recall tools
    registerAllTools({ api, config, logger });
  },
};

export default memoriPlugin;
