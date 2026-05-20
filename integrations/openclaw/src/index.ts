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

    // This code is temp. will be removed when https://github.com/openclaw/openclaw/issues/71428 is fixed.
    // allowConversationAccess was added to the schema in 2026.4.29.
    // However, during the 5.2/5.3 package refactor, origin detection was bugged (allowing
    // bypasses) and the schema validator temporarily rejected the key, crashing the app.
    // The platform stabilized the plugin loader in 2026.5.7 and re-clarified the rule in 5.9.
    // Therefore, we only enforce the warning on 2026.5.7 and newer.
    const ocVersion = api.runtime.version.replace(/^v/, '');
    const [ocYear = 0, ocMonth = 0, ocDay = 0] = ocVersion
      .split('.')
      .map((s) => parseInt(s, 10) || 0);

    const needsAccessPermission =
      ocYear > 2026 ||
      (ocYear === 2026 && ocMonth > 5) ||
      (ocYear === 2026 && ocMonth === 5 && ocDay >= 7);

    const hasConversationAccess =
      !needsAccessPermission ||
      api.config.plugins?.entries?.[PLUGIN_CONFIG.ID]?.hooks?.allowConversationAccess === true;

    // B: log at startup so the operator sees it immediately in gateway logs
    if (!hasConversationAccess) {
      api.logger.warn(
        `${PLUGIN_CONFIG.LOG_PREFIX} Conversation access is not enabled — agent_end memory capture will not work.\n` +
          `  To fix, run:\n` +
          `    openclaw config set plugins.entries.${PLUGIN_CONFIG.ID}.hooks.allowConversationAccess true\n` +
          `  Then restart the gateway.`
      );
    }

    const configContext = `Memori plugin configuration: projectId="${config.projectId}", entityId="${config.entityId}"`;

    const accessWarning = !hasConversationAccess
      ? [
          '⚠️ MEMORI SETUP REQUIRED — respond to this before anything else.',
          'Memory capture is disabled because conversation access is not enabled.',
          'Tell the user this now:',
          `  "Memori is installed but not fully set up. Run this command then restart the gateway:"`,
          `  openclaw config set plugins.entries.${PLUGIN_CONFIG.ID}.hooks.allowConversationAccess true`,
        ].join('\n')
      : null;

    // Static skills + config go in appendSystemContext (cached).
    // The access warning uses prependContext so it lands above the user's message
    // on every turn — the model reads it before any user input.
    api.on('before_prompt_build', () => ({
      appendSystemContext: [skillsContent, configContext].filter(Boolean).join('\n\n'),
      ...(accessWarning ? { prependContext: accessWarning } : {}),
    }));

    if (hasConversationAccess) {
      api.on('agent_end', (event: unknown, ctx: unknown) =>
        handleAugmentation(event as OpenClawEvent, ctx as OpenClawContext, config, logger)
      );
    }

    registerAuthenticatedTools({ api, config, logger });
  },
};

export default memoriPlugin;
