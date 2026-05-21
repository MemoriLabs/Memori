import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import { createRequire } from 'module';
import type { ToolDeps } from './types.js';

const execAsync = promisify(exec);
const require = createRequire(import.meta.url);

export function createMemoriQuotaTool(deps: ToolDeps) {
  const { logger, config } = deps;

  return {
    name: 'memori_quota',
    label: 'Memori Quota',
    description:
      'Retrieves the current memory usage and maximum allowed quota for the user. Use this when the user asks about their limits, storage, or how many memories they have left — or when you encounter errors suggesting memory limits have been reached and want to confirm before degrading behavior.',
    parameters: {
      type: 'object',
      properties: {},
    },

    async execute(_toolCallId: string) {
      try {
        logger.info('memori_quota checking usage...');

        const sdkEntry = require.resolve('@memorilabs/memori');
        const binPath = path.join(path.dirname(sdkEntry), 'bin/cli.js');
        const { stdout } = await execAsync(`${process.execPath} ${binPath} quota`, {
          env: { ...process.env, MEMORI_API_KEY: config.apiKey },
        });

        const result = {
          success: true,
          message: stdout.trim(),
        };

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          details: null,
        };
      } catch (e: unknown) {
        logger.warn(`memori_quota CLI failed: ${String(e)}`);

        let output = 'An unexpected error occurred while trying to fetch quota via the CLI.';

        if (typeof e === 'object' && e !== null) {
          const errObj = e as Record<string, unknown>;

          const stdout = typeof errObj.stdout === 'string' ? errObj.stdout.trim() : '';
          const stderr = typeof errObj.stderr === 'string' ? errObj.stderr.trim() : '';
          const msg = typeof errObj.message === 'string' ? errObj.message : '';

          output = stdout || stderr || msg || output;
        } else if (typeof e === 'string') {
          output = e;
        }

        const errorResult = {
          error: output,
        };

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
          details: null,
        };
      }
    },
  };
}
