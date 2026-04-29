import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('os', () => ({
  homedir: vi.fn(() => '/mock/home'),
}));

vi.mock('fs', () => ({
  existsSync: vi.fn(),
  mkdirSync: vi.fn(),
  readFileSync: vi.fn(),
  writeFileSync: vi.fn(),
}));

import { readPluginConfig, writePluginConfig, CONFIG_KEY_MAP } from '../../src/cli/config-file.js';

const PLUGIN_ID = 'openclaw-memori';

function makeFullConfig(cfg: Record<string, unknown>) {
  return {
    plugins: {
      entries: {
        [PLUGIN_ID]: { config: cfg },
      },
    },
  };
}

describe('cli/config-file', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('CONFIG_KEY_MAP', () => {
    it('should map api-key to apiKey', () => {
      expect(CONFIG_KEY_MAP['api-key']).toBe('apiKey');
    });

    it('should map entity-id to entityId', () => {
      expect(CONFIG_KEY_MAP['entity-id']).toBe('entityId');
    });

    it('should map project-id to projectId', () => {
      expect(CONFIG_KEY_MAP['project-id']).toBe('projectId');
    });
  });

  describe('readPluginConfig', () => {
    it('should return empty object when config file does not exist', async () => {
      const { existsSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(false);

      expect(readPluginConfig()).toEqual({});
    });

    it('should return all three fields when present in config file', async () => {
      const { existsSync, readFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue(
        JSON.stringify(
          makeFullConfig({ apiKey: 'my-key', entityId: 'my-entity', projectId: 'my-project' })
        )
      );

      expect(readPluginConfig()).toEqual({
        apiKey: 'my-key',
        entityId: 'my-entity',
        projectId: 'my-project',
      });
    });

    it('should return empty object when plugin entry is absent', async () => {
      const { existsSync, readFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue(JSON.stringify({ plugins: { entries: {} } }));

      expect(readPluginConfig()).toEqual({});
    });

    it('should return empty object when plugins key is absent', async () => {
      const { existsSync, readFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue(JSON.stringify({}));

      expect(readPluginConfig()).toEqual({});
    });

    it('should throw when the config file contains invalid JSON', async () => {
      const { existsSync, readFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue('{ not valid json }');

      expect(() => readPluginConfig()).toThrow('Failed to parse OpenClaw config');
    });
  });

  describe('writePluginConfig', () => {
    it('should create directory and write config when file does not exist', async () => {
      const { existsSync, mkdirSync, writeFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(false);

      writePluginConfig({ apiKey: 'new-key', entityId: 'new-entity', projectId: 'new-project' });

      expect(mkdirSync).toHaveBeenCalledWith(expect.stringContaining('.openclaw'), {
        recursive: true,
      });
      const written = JSON.parse(vi.mocked(writeFileSync).mock.calls[0][1] as string);
      expect(written.plugins.entries[PLUGIN_ID].config).toEqual({
        apiKey: 'new-key',
        entityId: 'new-entity',
        projectId: 'new-project',
      });
    });

    it('should merge new values into existing config without overwriting unrelated fields', async () => {
      const { existsSync, readFileSync, writeFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(true);
      vi.mocked(readFileSync).mockReturnValue(
        JSON.stringify(makeFullConfig({ apiKey: 'existing-key', entityId: 'existing-entity' }))
      );

      writePluginConfig({ projectId: 'added-project' });

      const written = JSON.parse(vi.mocked(writeFileSync).mock.calls[0][1] as string);
      const cfg = written.plugins.entries[PLUGIN_ID].config;
      expect(cfg.apiKey).toBe('existing-key');
      expect(cfg.entityId).toBe('existing-entity');
      expect(cfg.projectId).toBe('added-project');
    });

    it('should skip falsy values and not write them', async () => {
      const { existsSync, writeFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(false);

      writePluginConfig({ apiKey: 'real-key', entityId: '' });

      const written = JSON.parse(vi.mocked(writeFileSync).mock.calls[0][1] as string);
      const cfg = written.plugins.entries[PLUGIN_ID].config;
      expect(cfg.apiKey).toBe('real-key');
      expect(cfg).not.toHaveProperty('entityId');
    });

    it('should append a newline to the written JSON', async () => {
      const { existsSync, writeFileSync } = await import('fs');
      vi.mocked(existsSync).mockReturnValue(false);

      writePluginConfig({ apiKey: 'key' });

      const content = vi.mocked(writeFileSync).mock.calls[0][1] as string;
      expect(content.endsWith('\n')).toBe(true);
    });
  });
});
