import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { homedir } from 'os';
import { dirname, join } from 'path';

const PLUGIN_ID = 'openclaw-memori';
const CONFIG_PATH = join(homedir(), '.openclaw', 'openclaw.json');

export interface MemoriCLIConfig {
  apiKey?: string;
  entityId?: string;
  projectId?: string;
}

function readFullConfig(): Record<string, unknown> {
  if (!existsSync(CONFIG_PATH)) return {};
  try {
    return JSON.parse(readFileSync(CONFIG_PATH, 'utf-8')) as Record<string, unknown>;
  } catch (e) {
    throw new Error(`Failed to parse OpenClaw config at ${CONFIG_PATH}: ${String(e)}`);
  }
}

function writeFullConfig(config: Record<string, unknown>): void {
  mkdirSync(dirname(CONFIG_PATH), { recursive: true });
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + '\n', 'utf-8');
}

function ensurePluginStructure(config: Record<string, unknown>): void {
  if (!config.plugins || typeof config.plugins !== 'object') config.plugins = {};
  const plugins = config.plugins as Record<string, unknown>;
  if (!plugins.entries || typeof plugins.entries !== 'object') plugins.entries = {};
  const entries = plugins.entries as Record<string, unknown>;
  if (!entries[PLUGIN_ID] || typeof entries[PLUGIN_ID] !== 'object') entries[PLUGIN_ID] = {};
  const entry = entries[PLUGIN_ID] as Record<string, unknown>;
  if (!entry.config || typeof entry.config !== 'object') entry.config = {};
}

function getPluginConfigBlock(full: Record<string, unknown>): Record<string, unknown> | undefined {
  const plugins = full.plugins as Record<string, unknown> | undefined;
  const entries = plugins?.entries as Record<string, unknown> | undefined;
  const entry = entries?.[PLUGIN_ID] as Record<string, unknown> | undefined;
  return entry?.config as Record<string, unknown> | undefined;
}

export function readPluginConfig(): MemoriCLIConfig {
  const cfg = getPluginConfigBlock(readFullConfig());
  if (!cfg) return {};
  return {
    apiKey: cfg.apiKey as string | undefined,
    entityId: cfg.entityId as string | undefined,
    projectId: cfg.projectId as string | undefined,
  };
}

export function writePluginConfig(updates: Partial<MemoriCLIConfig>): void {
  const full = readFullConfig();
  ensurePluginStructure(full);
  const entry = ((full.plugins as Record<string, unknown>).entries as Record<string, unknown>)[
    PLUGIN_ID
  ] as Record<string, unknown>;
  const cfg = entry.config as Record<string, unknown>;
  for (const [key, value] of Object.entries(updates)) {
    if (value) cfg[key] = value;
  }
  writeFullConfig(full);
}

export const CONFIG_KEY_MAP: Record<string, keyof MemoriCLIConfig> = {
  'api-key': 'apiKey',
  'entity-id': 'entityId',
  'project-id': 'projectId',
};
