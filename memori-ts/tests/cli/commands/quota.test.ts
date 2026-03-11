import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { quotaCommand } from '../../../src/cli/commands/quota.js';
import * as utils from '../../../src/cli/utils.js';

vi.mock('../../../src/cli/utils.js', () => ({
  printBanner: vi.fn(),
}));

const mockGet = vi.fn();

vi.mock('../../../src/core/network.js', () => {
  return {
    Api: class MockApi {
      get = mockGet;
    },
  };
});

describe('quotaCommand', () => {
  let consoleLogSpy: ReturnType<typeof vi.spyOn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;
  let processExitSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    processExitSpy = vi.spyOn(process, 'exit').mockImplementation((() => {}) as any) as any;

    mockGet.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should fetch and print quota details successfully', async () => {
    mockGet.mockResolvedValue({
      memories: { max: 5000, num: 1234 },
      message: 'Quota is healthy',
    });

    await quotaCommand([]);

    expect(utils.printBanner).toHaveBeenCalled();
    expect(mockGet).toHaveBeenCalledWith('sdk/quota');

    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining('Maximum # of Memories: 5,000')
    );
    expect(consoleLogSpy).toHaveBeenCalledWith(
      expect.stringContaining('Current # of Memories: 1,234')
    );
    expect(consoleLogSpy).toHaveBeenCalledWith(expect.stringContaining('+ Quota is healthy'));

    expect(processExitSpy).not.toHaveBeenCalled();
  });

  it('should print error and exit with code 1 on API failure', async () => {
    mockGet.mockRejectedValue(new Error('Invalid API Key'));

    await quotaCommand([]);

    expect(consoleErrorSpy).toHaveBeenCalledWith(
      'Failed to fetch quota. Please check your MEMORI_API_KEY.'
    );
    expect(consoleErrorSpy).toHaveBeenCalledWith('Error details: Invalid API Key');

    expect(processExitSpy).toHaveBeenCalledWith(1);
  });
});
