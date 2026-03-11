import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { main } from '../../../src/cli/router.js';
import { quotaCommand } from '../../../src/cli/commands/quota.js';
import { helpCommand } from '../../../src/cli/commands/help.js';

vi.mock('../../../src/cli/commands/quota.js', () => ({
  quotaCommand: vi.fn(),
}));
vi.mock('../../../src/cli/commands/help.js', () => ({
  helpCommand: vi.fn(),
}));

describe('CLI Router', () => {
  let originalArgv: string[];
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;
  let processExitSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    originalArgv = process.argv;

    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any) as any;

    vi.clearAllMocks();
  });

  afterEach(() => {
    process.argv = originalArgv;
    vi.restoreAllMocks();
  });

  it('should route to the quota command successfully', async () => {
    process.argv = ['node', 'cli.js', 'quota'];
    await main();

    expect(quotaCommand).toHaveBeenCalledWith([]);
  });

  it('should route to the help command when no arguments are provided', async () => {
    process.argv = ['node', 'cli.js'];
    await main();

    expect(helpCommand).toHaveBeenCalled();
  });

  it('should print an error, show help, and exit on an unknown command', async () => {
    process.argv = ['node', 'cli.js', 'unknown-command'];
    await main();

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining("Unknown command 'unknown-command'")
    );
    expect(helpCommand).toHaveBeenCalled();
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });

  it('should catch unexpected errors, log them, and exit gracefully', async () => {
    vi.mocked(quotaCommand).mockRejectedValueOnce(new Error('Explosion!'));

    process.argv = ['node', 'cli.js', 'quota'];
    await main();

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      expect.stringContaining("Unexpected error executing command 'quota'")
    );
    expect(consoleErrorSpy).toHaveBeenCalledWith(expect.stringContaining('Explosion!'));
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });
});
