import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { helpCommand } from '../../../src/cli/commands/help.js';
import * as utils from '../../../src/cli/utils.js';

vi.mock('../../../src/cli/utils.js', () => ({
  printBanner: vi.fn(),
}));

describe('helpCommand', () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should print the banner and help message', async () => {
    await helpCommand([]);

    expect(utils.printBanner).toHaveBeenCalled();

    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Usage: memori <command>'));
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('Available Commands:'));
  });
});
