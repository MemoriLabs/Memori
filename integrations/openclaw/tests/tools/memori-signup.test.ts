import { describe, it, expect, vi, beforeEach } from 'vitest';
import { exec } from 'child_process';
import { createMemoriSignupTool } from '../../src/tools/memori-signup.js';
import type { ToolDeps } from '../../src/tools/types.js';

vi.mock('child_process', () => ({
  exec: vi.fn(),
}));

vi.mock('os', () => ({
  tmpdir: vi.fn(() => '/mock-tmp'),
}));

describe('tools/memori-signup', () => {
  const mockExec = vi.mocked(exec) as any;
  let deps: ToolDeps;

  beforeEach(() => {
    vi.clearAllMocks();

    mockExec.mockImplementation((_cmd: string, cb: Function) => {
      cb(null, { stdout: 'default output', stderr: '' });
      return {} as any;
    });

    deps = {
      api: {} as any,
      config: { apiKey: 'test-key', entityId: 'test-entity', projectId: 'default-project' },
      logger: {
        info: vi.fn(),
        warn: vi.fn(),
        error: vi.fn(),
        section: vi.fn(),
        endSection: vi.fn(),
      } as any,
    };
  });

  describe('tool definition', () => {
    it('should have the correct name', () => {
      expect(createMemoriSignupTool(deps).name).toBe('memori_signup');
    });

    it('should have the correct label', () => {
      expect(createMemoriSignupTool(deps).label).toBe('Memori Sign Up');
    });

    it('should have a description', () => {
      expect(createMemoriSignupTool(deps).description).toBeTruthy();
    });

    it('should define the email parameter as a string', () => {
      const { parameters } = createMemoriSignupTool(deps);
      expect((parameters.properties as any).email).toBeDefined();
      expect((parameters.properties as any).email.type).toBe('string');
    });

    it('should require the email parameter', () => {
      expect(createMemoriSignupTool(deps).parameters.required).toContain('email');
    });
  });

  describe('email validation', () => {
    const invalidEmails = [
      'notanemail',
      '@domain.com',
      'user@',
      '',
      'user@domain',
      'user @example.com',
      'user@.com',
    ];

    for (const email of invalidEmails) {
      it(`should reject invalid email: "${email}"`, async () => {
        const tool = createMemoriSignupTool(deps);
        const result = await tool.execute('call-1', { email });
        expect(JSON.parse(result.content[0].text)).toHaveProperty('error');
        expect(mockExec).not.toHaveBeenCalled();
      });
    }

    it('should include the invalid email in the error message', async () => {
      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'bademail' });
      expect(JSON.parse(result.content[0].text).error).toContain('bademail');
    });

    it('should log a warning for an invalid email', async () => {
      const tool = createMemoriSignupTool(deps);
      await tool.execute('call-1', { email: 'bad' });
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_signup rejected email format')
      );
    });

    it('should accept a standard email', async () => {
      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text)).toHaveProperty('success', true);
    });

    it('should accept an email with subdomains', async () => {
      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@mail.example.co.uk' });
      expect(JSON.parse(result.content[0].text)).toHaveProperty('success', true);
    });

    it('should accept an email with plus addressing', async () => {
      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user+tag@example.com' });
      expect(JSON.parse(result.content[0].text)).toHaveProperty('success', true);
    });
  });

  describe('execute', () => {
    it('should install the package into the tmp directory', async () => {
      const tool = createMemoriSignupTool(deps);
      await tool.execute('call-1', { email: 'user@example.com' });
      expect(mockExec.mock.calls[0][0]).toContain('npm install --prefix /mock-tmp');
      expect(mockExec.mock.calls[0][0]).toContain('@memorilabs/memori');
    });

    it('should run sign-up using the explicit binary path in tmp', async () => {
      const tool = createMemoriSignupTool(deps);
      await tool.execute('call-1', { email: 'user@example.com' });
      expect(mockExec.mock.calls[1][0]).toContain('/mock-tmp/node_modules/.bin/memori');
      expect(mockExec.mock.calls[1][0]).toContain('sign-up user@example.com');
    });

    it('should return success: true with trimmed CLI stdout', async () => {
      mockExec
        .mockImplementationOnce((_cmd: string, cb: Function) => {
          cb(null, { stdout: '', stderr: '' });
          return {} as any;
        })
        .mockImplementationOnce((_cmd: string, cb: Function) => {
          cb(null, { stdout: '  Your key is on the way!  \n', stderr: '' });
          return {} as any;
        });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text)).toEqual({
        success: true,
        message: 'Your key is on the way!',
      });
    });

    it('should return details: null on success', async () => {
      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(result.details).toBeNull();
    });

    it('should log info with the email before executing', async () => {
      const tool = createMemoriSignupTool(deps);
      await tool.execute('call-1', { email: 'user@example.com' });
      expect(deps.logger.info).toHaveBeenCalledWith(expect.stringContaining('user@example.com'));
    });

    it('should call exec exactly twice on success', async () => {
      const tool = createMemoriSignupTool(deps);
      await tool.execute('call-1', { email: 'user@example.com' });
      expect(mockExec).toHaveBeenCalledTimes(2);
    });
  });

  describe('error handling', () => {
    it('should return error JSON and log warn on exec failure', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb(new Error('exec failed'));
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text)).toHaveProperty('error');
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_signup CLI failed')
      );
    });

    it('should use error.stdout when present', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb(Object.assign(new Error(), { stdout: 'stdout detail', stderr: '' }));
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text).error).toBe('stdout detail');
    });

    it('should fall back to error.stderr when stdout is empty', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb(Object.assign(new Error(), { stdout: '', stderr: 'stderr detail' }));
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text).error).toBe('stderr detail');
    });

    it('should fall back to error.message when stdout and stderr are empty', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb(Object.assign(new Error('message detail'), { stdout: '', stderr: '' }));
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text).error).toBe('message detail');
    });

    it('should use a default message when the error has no useful fields', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb({});
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text).error).toContain('unexpected error');
    });

    it('should handle a plain string error', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb('plain string error');
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text).error).toBe('plain string error');
    });

    it('should trim whitespace from error.stdout', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb(Object.assign(new Error(), { stdout: '  trimmed error  \n', stderr: '' }));
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(JSON.parse(result.content[0].text).error).toBe('trimmed error');
    });

    it('should return details: null on error', async () => {
      mockExec.mockImplementation((_cmd: string, cb: Function) => {
        cb(new Error('failed'));
        return {} as any;
      });

      const tool = createMemoriSignupTool(deps);
      const result = await tool.execute('call-1', { email: 'user@example.com' });
      expect(result.details).toBeNull();
    });
  });
});
