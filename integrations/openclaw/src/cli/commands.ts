import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { createRecallClient } from '../utils/memori-client.js';
import { CONFIG_KEY_MAP, readPluginConfig, writePluginConfig } from './config-file.js';

function maskKey(key: string): string {
  if (key.length <= 8) return '****';
  return `****...${key.slice(-4)}`;
}

function isReady(cfg: ReturnType<typeof readPluginConfig>): boolean {
  return Boolean(cfg.apiKey && cfg.entityId && cfg.projectId);
}

export function registerCliCommands(api: OpenClawPluginApi): void {
  api.registerCli(
    ({ program }) => {
      const memori = program
        .command('memori')
        .description('Memori memory plugin commands')
        .configureHelp({ sortSubcommands: true });

      // ── init ────────────────────────────────────────────────────────────────
      memori
        .command('init')
        .description('Configure the Memori plugin with your API credentials')
        .option('--api-key <key>', 'Memori API key (from app.memorilabs.ai)')
        .option('--entity-id <id>', 'Entity ID to scope all memories to')
        .option('--project-id <id>', 'Project ID to scope all memories to')
        .action((opts: { apiKey?: string; entityId?: string; projectId?: string }) => {
          const missing: string[] = [];
          if (!opts.apiKey) missing.push('--api-key');
          if (!opts.entityId) missing.push('--entity-id');
          if (!opts.projectId) missing.push('--project-id');

          if (missing.length > 0) {
            console.error(`Error: missing required option(s): ${missing.join(', ')}`);
            console.error(
              '\nUsage:\n  openclaw memori init --api-key <key> --entity-id <id> --project-id <id>'
            );
            process.exitCode = 1;
            return;
          }

          writePluginConfig({
            apiKey: opts.apiKey,
            entityId: opts.entityId,
            projectId: opts.projectId,
          });

          console.log('\nMemori configured successfully.\n');
          console.log(`  API Key:    ${maskKey(opts.apiKey || '')}`);
          console.log(`  Entity ID:  ${opts.entityId}`);
          console.log(`  Project ID: ${opts.projectId}`);
          console.log('\nRestart the gateway to apply changes:');
          console.log('  openclaw gateway restart\n');
        });

      // ── status ──────────────────────────────────────────────────────────────
      memori
        .command('status')
        .description('Show current configuration and verify API connectivity')
        .option('--check', 'Test live API connectivity')
        .action(async (opts: { check?: boolean }) => {
          const cfg = readPluginConfig();
          const ready = isReady(cfg);

          console.log('\nMemori Plugin Status');
          console.log('─'.repeat(36));
          console.log(`  API Key:    ${cfg.apiKey ? maskKey(cfg.apiKey) : '(not set)'}`);
          console.log(`  Entity ID:  ${cfg.entityId ?? '(not set)'}`);
          console.log(`  Project ID: ${cfg.projectId ?? '(not set)'}`);
          console.log();

          if (!ready) {
            console.log('Status: Not configured.');
            console.log(
              'Run:    openclaw memori init --api-key <key> --entity-id <id> --project-id <id>\n'
            );
            process.exitCode = 1;
            return;
          }

          if (opts.check) {
            process.stdout.write('Checking API connectivity... ');
            try {
              const client = createRecallClient(cfg.apiKey || '', cfg.entityId || '');
              await client.agentRecall({ projectId: cfg.projectId });
              console.log('OK');
            } catch (e) {
              console.log('FAILED');
              console.error(`  ${String(e)}\n`);
              process.exitCode = 1;
              return;
            }
          }

          console.log('Status: Ready\n');
        });

      // ── config ──────────────────────────────────────────────────────────────
      const configCmd = memori
        .command('config')
        .description('Manage plugin configuration')
        .configureHelp({ sortSubcommands: true });

      const KEYS = Object.keys(CONFIG_KEY_MAP).join(', ');

      configCmd
        .command('show')
        .description('Display current configuration')
        .action(() => {
          const cfg = readPluginConfig();
          console.log('\nMemori Plugin Configuration');
          console.log('─'.repeat(36));
          console.log(`  api-key:    ${cfg.apiKey ? maskKey(cfg.apiKey) : '(not set)'}`);
          console.log(`  entity-id:  ${cfg.entityId ?? '(not set)'}`);
          console.log(`  project-id: ${cfg.projectId ?? '(not set)'}`);
          console.log();
        });

      configCmd
        .command('get')
        .argument('<key>', `Config key — one of: ${KEYS}`)
        .description('Get a specific config value')
        .action((key: string) => {
          if (!(key in CONFIG_KEY_MAP)) {
            console.error(`Unknown key "${key}". Valid keys: ${KEYS}`);
            process.exitCode = 1;
            return;
          }
          const field = CONFIG_KEY_MAP[key];
          const value = readPluginConfig()[field];
          if (value === undefined) {
            console.log('(not set)');
          } else {
            console.log(field === 'apiKey' ? maskKey(value) : value);
          }
        });

      configCmd
        .command('set')
        .argument('<key>', `Config key — one of: ${KEYS}`)
        .argument('<value>', 'Value to set')
        .description('Set a specific config value')
        .action((key: string, value: string) => {
          if (!(key in CONFIG_KEY_MAP)) {
            console.error(`Unknown key "${key}". Valid keys: ${KEYS}`);
            process.exitCode = 1;
            return;
          }
          const field = CONFIG_KEY_MAP[key];
          writePluginConfig({ [field]: value });
          console.log(`Set ${key}.`);
        });
    },
    {
      descriptors: [
        { name: 'memori', description: 'Memori memory plugin commands', hasSubcommands: true },
      ],
    }
  );
}
