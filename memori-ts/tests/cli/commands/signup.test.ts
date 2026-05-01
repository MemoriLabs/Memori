import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { signupCommand } from '../../../src/cli/commands/signup.js';
import * as utils from '../../../src/cli/utils.js';
import { MemoriApiValidationError } from '../../../src/core/errors.js';

vi.mock('../../../src/cli/utils.js', () => ({
  printBanner: vi.fn(),
}));

const mockPost = vi.fn();

vi.mock('../../../src/core/network.js', () => {
  return {
    Api: class MockApi {
      post = mockPost;
    },
  };
});

describe('signupCommand', () => {
  let consoleLogSpy: ReturnType<typeof vi.spyOn>;
  let consoleErrorSpy: ReturnType<typeof vi.spyOn>;
  let processExitSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    processExitSpy = vi.spyOn(process, 'exit').mockImplementation(((code?: number) => {
      throw new Error(`process.exit called with code: ${code}`);
    }) as any) as any;

    mockPost.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should print usage and exit if no email is provided', async () => {
    // Expect the command to reject with our mock exit error
    await expect(signupCommand([])).rejects.toThrow('process.exit called with code: 1');

    expect(utils.printBanner).toHaveBeenCalled();
    expect(consoleLogSpy).toHaveBeenCalledWith('Usage: memori sign-up <email_address>\n');
    expect(processExitSpy).toHaveBeenCalledWith(1);
    expect(mockPost).not.toHaveBeenCalled();
  });

  it('should post email and print returned content on success', async () => {
    mockPost.mockResolvedValue({ content: 'Custom API success message!' });

    await signupCommand(['test@example.com']);

    expect(mockPost).toHaveBeenCalledWith('sdk/account', { email: 'test@example.com' });
    expect(consoleLogSpy).toHaveBeenCalledWith('Custom API success message!');
    expect(processExitSpy).not.toHaveBeenCalled();
  });

  it('should post email and print fallback content on success if none is returned', async () => {
    mockPost.mockResolvedValue({}); // No content field

    await signupCommand(['test@example.com']);

    expect(consoleLogSpy).toHaveBeenCalledWith("You're all set! We sent you an email.\n");
    expect(processExitSpy).not.toHaveBeenCalled();
  });

  it('should gracefully handle validation errors and inform the user', async () => {
    mockPost.mockRejectedValue(new MemoriApiValidationError(422, 'Email is invalid format'));

    await signupCommand(['bad-email']);

    expect(consoleLogSpy).toHaveBeenCalledWith(
      'The email you provided "bad-email" is not valid.\n'
    );
    expect(processExitSpy).not.toHaveBeenCalled();
  });

  it('should print error and exit with code 1 on unexpected API failure', async () => {
    mockPost.mockRejectedValue(new Error('Network timeout'));

    // Expect the command to reject with our mock exit error
    await expect(signupCommand(['test@example.com'])).rejects.toThrow(
      'process.exit called with code: 1'
    );

    expect(consoleErrorSpy).toHaveBeenCalledWith('An unexpected error occurred during sign-up.');
    expect(consoleErrorSpy).toHaveBeenCalledWith('Error details: Network timeout\n');
    expect(processExitSpy).toHaveBeenCalledWith(1);
  });
});
