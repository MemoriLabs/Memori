import { exec } from 'child_process';
import { promisify } from 'util';
import * as os from 'os';
import * as path from 'path';
import type { ToolDeps } from './types.js';

const execAsync = promisify(exec);

export function createMemoriSignupTool(deps: ToolDeps) {
  const { logger } = deps;

  return {
    name: 'memori_signup',
    label: 'Memori Sign Up',
    description:
      'CRITICAL: You MUST use this tool when the user asks to sign up, create an account, or get an API key for Memori — or when you encounter a missing MEMORI_API_KEY error and the user provides their email. If the user has not provided an email address, ask for it first. Do not guess or hallucinate an email.',
    parameters: {
      type: 'object',
      properties: {
        email: {
          type: 'string',
          description: 'The email address to send the Memori API key to.',
        },
      },
      required: ['email'],
    },

    async execute(
      _toolCallId: string,
      params: {
        email: string;
      }
    ) {
      try {
        const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
        if (!emailRegex.test(params.email)) {
          const errorResult = {
            error: `The email you provided "${params.email}" is not valid. Please provide a standard email address.`,
          };
          logger.warn(`memori_signup rejected email format: ${params.email}`);
          return {
            content: [{ type: 'text' as const, text: JSON.stringify(errorResult) }],
            details: null,
          };
        }

        logger.info(`memori_signup attempting to sign up: ${params.email}`);

        const tmpDir = os.tmpdir();

        await execAsync(`npm install --prefix ${tmpDir} --no-save @memorilabs/memori@0.1.12-beta`);

        const binPath = path.join(tmpDir, 'node_modules', '.bin', 'memori');
        const { stdout } = await execAsync(`${binPath} sign-up ${params.email}`);

        const result = {
          success: true,
          message: stdout.trim(),
        };

        return {
          content: [{ type: 'text' as const, text: JSON.stringify(result) }],
          details: null,
        };
      } catch (e: unknown) {
        logger.warn(`memori_signup CLI failed: ${String(e)}`);

        let output = 'An unexpected error occurred while trying to sign up via the CLI.';

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
