import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { printBanner } from '../../src/cli/utils.js';

describe('CLI Utils', () => {
  let consoleSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should print the ASCII banner successfully', () => {
    printBanner();

    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('perfectam memoriam'));
    expect(consoleSpy).toHaveBeenCalledWith(expect.stringContaining('memorilabs.ai'));
  });
});
