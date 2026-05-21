import { describe, it, expect, vi, beforeEach } from 'vitest';
import { exec } from 'child_process';
import { createMemoriQuotaTool } from '../../src/tools/memori-quota.js';
import type { ToolDeps } from '../../src/tools/types.js';

vi.mock('child_process', () => ({
  exec: vi.fn(),
}));

describe('tools/memori-quota', () => {
  const mockExec = vi.mocked(exec) as any;
  let deps: ToolDeps;

  beforeEach(() => {
    vi.clearAllMocks();

    mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
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
      expect(createMemoriQuotaTool(deps).name).toBe('memori_quota');
    });

    it('should have the correct label', () => {
      expect(createMemoriQuotaTool(deps).label).toBe('Memori Quota');
    });

    it('should have a description', () => {
      expect(createMemoriQuotaTool(deps).description).toBeTruthy();
    });

    it('should have no required parameters', () => {
      const { parameters } = createMemoriQuotaTool(deps);
      expect((parameters as any).required).toBeUndefined();
    });

    it('should have no parameter properties defined', () => {
      const { parameters } = createMemoriQuotaTool(deps);
      expect(Object.keys(parameters.properties)).toHaveLength(0);
    });
  });

  describe('execute', () => {
    it('should run the quota command via the local SDK binary', async () => {
      const tool = createMemoriQuotaTool(deps);
      await tool.execute('call-1');
      const cmd: string = mockExec.mock.calls[0][0];
      expect(cmd).toContain('@memorilabs/memori');
      expect(cmd).toContain('quota');
    });

    it('should invoke the binary with node and the cli.js path', async () => {
      const tool = createMemoriQuotaTool(deps);
      await tool.execute('call-1');
      const cmd: string = mockExec.mock.calls[0][0];
      expect(cmd).toContain(process.execPath);
      expect(cmd).toContain('cli.js');
    });

    it('should pass the API key in env', async () => {
      const tool = createMemoriQuotaTool(deps);
      await tool.execute('call-1');
      const options = mockExec.mock.calls[0][1];
      expect(options.env.MEMORI_API_KEY).toBe('test-key');
    });

    it('should return success: true with trimmed CLI stdout', async () => {
      mockExec.mockImplementationOnce((_cmd: string, _options: any, cb: Function) => {
        cb(null, { stdout: '  Used: 42 / 1000  \n', stderr: '' });
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text)).toEqual({
        success: true,
        message: 'Used: 42 / 1000',
      });
    });

    it('should return details: null on success', async () => {
      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(result.details).toBeNull();
    });

    it('should log info before executing', async () => {
      const tool = createMemoriQuotaTool(deps);
      await tool.execute('call-1');
      expect(deps.logger.info).toHaveBeenCalledWith(expect.stringContaining('memori_quota'));
    });

    it('should call exec exactly once on success', async () => {
      const tool = createMemoriQuotaTool(deps);
      await tool.execute('call-1');
      expect(mockExec).toHaveBeenCalledTimes(1);
    });
  });

  describe('error handling', () => {
    it('should return error JSON and log warn on exec failure', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb(new Error('exec failed'));
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text)).toHaveProperty('error');
      expect(deps.logger.warn).toHaveBeenCalledWith(
        expect.stringContaining('memori_quota CLI failed')
      );
    });

    it('should use error.stdout when present', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb(Object.assign(new Error(), { stdout: 'stdout detail', stderr: '' }));
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text).error).toBe('stdout detail');
    });

    it('should fall back to error.stderr when stdout is empty', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb(Object.assign(new Error(), { stdout: '', stderr: 'stderr detail' }));
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text).error).toBe('stderr detail');
    });

    it('should fall back to error.message when stdout and stderr are empty', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb(Object.assign(new Error('message detail'), { stdout: '', stderr: '' }));
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text).error).toBe('message detail');
    });

    it('should use a default message when the error has no useful fields', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb({});
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text).error).toContain('unexpected error');
    });

    it('should handle a plain string error', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb('plain string error');
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text).error).toBe('plain string error');
    });

    it('should trim whitespace from error.stdout', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb(Object.assign(new Error(), { stdout: '  trimmed error  \n', stderr: '' }));
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(JSON.parse(result.content[0].text).error).toBe('trimmed error');
    });

    it('should return details: null on error', async () => {
      mockExec.mockImplementation((_cmd: string, _options: any, cb: Function) => {
        cb(new Error('failed'));
        return {} as any;
      });

      const tool = createMemoriQuotaTool(deps);
      const result = await tool.execute('call-1');
      expect(result.details).toBeNull();
    });
  });
});
