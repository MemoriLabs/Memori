import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import type { MemoriPluginConfig } from '../types.js';
import type { MemoriLogger } from '../utils/logger.js';

export interface ToolDeps {
  api: OpenClawPluginApi;
  config: MemoriPluginConfig;
  logger: MemoriLogger;
}
