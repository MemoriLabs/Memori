import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { handleAugmentation } from './handlers/augmentation.js';
import { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from './types.js';
import { PLUGIN_CONFIG } from './constants.js';
import { MemoriLogger, loadSkillsContent } from './utils/index.js';
import { registerUtilityTools, registerAuthenticatedTools } from './tools/index.js';
import { registerCliCommands } from './cli/commands.js';

const memoriPlugin = {
  id: PLUGIN_CONFIG.ID,
  name: PLUGIN_CONFIG.NAME,
  description: 'Hosted memory backend',

  register(api: OpenClawPluginApi) {
    registerCliCommands(api);

    const rawConfig = api.pluginConfig;

    const config: MemoriPluginConfig = {
      apiKey: rawConfig?.apiKey as string,
      entityId: rawConfig?.entityId as string,
      projectId: rawConfig?.projectId as string,
    };

    const logger = new MemoriLogger(api);
    const skillsContent = loadSkillsContent(api.resolvePath.bind(api));

    registerUtilityTools({ api, config, logger });

    if (!config.apiKey || !config.entityId) {
      api.logger.warn(
        `${PLUGIN_CONFIG.LOG_PREFIX} Missing apiKey or entityId in config. Plugin disabled.`
      );
      return;
    }

    logger.info(`\n=== ${PLUGIN_CONFIG.LOG_PREFIX} INITIALIZING PLUGIN ===`);
    logger.info(`${PLUGIN_CONFIG.LOG_PREFIX} Tracking Entity ID: ${config.entityId}`);

    const configContext = `Memori plugin configuration: projectId="${config.projectId}", entityId="${config.entityId}"`;
    api.on('before_prompt_build', () => ({
      appendSystemContext: [skillsContent, configContext].filter(Boolean).join('\n\n'),
    }));

    api.on('agent_end', (event: unknown, ctx: unknown) =>
      handleAugmentation(event as OpenClawEvent, ctx as OpenClawContext, config, logger)
    );

    registerAuthenticatedTools({ api, config, logger });
  },
};

export default memoriPlugin;
