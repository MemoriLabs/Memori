import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const { mainMock } = vi.hoisted(() => ({
  mainMock: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../../../src/cli/router.js', () => ({
  main: mainMock,
}));

describe('CLI bin', () => {
  let originalCwd: string;
  let originalApiKey: string | undefined;
  let tempDir: string;

  beforeEach(() => {
    originalCwd = process.cwd();
    originalApiKey = process.env.MEMORI_API_KEY;
    tempDir = mkdtempSync(join(tmpdir(), 'memori-cli-'));
    delete process.env.MEMORI_API_KEY;
    mainMock.mockClear();
    vi.resetModules();
  });

  afterEach(() => {
    process.chdir(originalCwd);
    if (originalApiKey === undefined) {
      delete process.env.MEMORI_API_KEY;
    } else {
      process.env.MEMORI_API_KEY = originalApiKey;
    }
    rmSync(tempDir, { recursive: true, force: true });
    vi.restoreAllMocks();
  });

  it('loads MEMORI_API_KEY from .env in the current working directory', async () => {
    writeFileSync(join(tempDir, '.env'), 'MEMORI_API_KEY=dotenv-key\n');
    process.chdir(tempDir);

    await import('../../../src/bin/cli.js');

    expect(process.env.MEMORI_API_KEY).toBe('dotenv-key');
    expect(mainMock).toHaveBeenCalledOnce();
  });

  it('does not override an exported MEMORI_API_KEY with .env', async () => {
    writeFileSync(join(tempDir, '.env'), 'MEMORI_API_KEY=dotenv-key\n');
    process.env.MEMORI_API_KEY = 'exported-key';
    process.chdir(tempDir);

    await import('../../../src/bin/cli.js');

    expect(process.env.MEMORI_API_KEY).toBe('exported-key');
    expect(mainMock).toHaveBeenCalledOnce();
  });
});
