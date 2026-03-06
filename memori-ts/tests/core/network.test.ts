import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Api, ApiSubdomain } from '../../src/core/network.js';
import { Config } from '../../src/core/config.js';
import {
  MemoriApiClientError,
  MemoriApiValidationError,
  QuotaExceededError,
  TimeoutError,
} from '../../src/core/errors.js';

// Mock the global fetch
const fetchMock = vi.fn();
global.fetch = fetchMock;

describe('Api Class', () => {
  let config: Config;
  let api: Api;

  beforeEach(() => {
    fetchMock.mockReset();
    vi.useFakeTimers();

    config = new Config();
    config.apiKey = 'test-api-key';
    config.baseUrl = 'https://api.test.com';
    config.timeout = 5000;
    config.testMode = true;

    api = new Api(config);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.useRealTimers();
  });

  it('should initialize with the correct collector subdomain', () => {
    const collectorApi = new Api(config, ApiSubdomain.COLLECTOR);
    expect(collectorApi).toBeDefined();
  });

  it('should make a successful GET request', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ data: 'success' }),
    });

    const result = await api.get<{ data: string }>('test-route');

    expect(result).toEqual({ data: 'success' });
    expect(fetchMock).toHaveBeenCalledWith(
      'https://api.test.com/v1/test-route',
      expect.objectContaining({
        method: 'GET',
        headers: expect.objectContaining({
          Authorization: 'Bearer test-api-key',
          'Content-Type': 'application/json',
        }),
      })
    );
  });

  it('should handle 204 No Content', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 204,
      json: async () => ({}),
    });

    const result = await api.post('test-route');
    expect(result).toEqual({});
  });

  it('should throw QuotaExceededError on 429', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 429,
      json: async () => ({ message: 'Rate limit exceeded' }),
    });

    await expect(api.get('test')).rejects.toThrow(QuotaExceededError);
  });

  it('should throw MemoriApiValidationError on 422', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 422,
      json: async () => ({ detail: 'Invalid input' }),
    });

    await expect(api.post('test', {})).rejects.toThrow(MemoriApiValidationError);
  });

  it('should retry on 5xx errors and eventually fail', async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 502,
      json: async () => ({ message: 'Server Error' }),
    });

    const promise = api.get('test');

    // Attach expectation BEFORE advancing timers to catch the rejection as it happens
    const assertion = expect(promise).rejects.toThrow(MemoriApiClientError);

    // Advance time enough to cover 5 retries with exponential backoff
    await vi.advanceTimersByTimeAsync(40000);

    await assertion;
    expect(fetchMock).toHaveBeenCalledTimes(6);
  });

  it('should handle network timeouts and retry until failure', async () => {
    config.timeout = 1000;

    fetchMock.mockImplementation((_url, options) => {
      const signal = options.signal;
      return new Promise((resolve, reject) => {
        if (signal) {
          signal.addEventListener('abort', () => {
            const err = new Error('Aborted');
            err.name = 'AbortError';
            reject(err);
          });
        }
      });
    });

    const promise = api.get('test');

    // Attach expectation BEFORE advancing timers
    const assertion = expect(promise).rejects.toThrow(TimeoutError);

    // Retry loop: Initial + 5 Retries.
    // Each has 1000ms timeout + backoff delays.
    // Total duration needed is > 30s.
    await vi.advanceTimersByTimeAsync(40000);

    await assertion;
    expect(fetchMock).toHaveBeenCalledTimes(6);
  });

  it('should make a successful PATCH request', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ updated: true }),
    });

    const result = await api.patch('test-route', { key: 'value' });
    expect(result).toEqual({ updated: true });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('test-route'),
      expect.objectContaining({ method: 'PATCH' })
    );
  });

  it('should make a successful DELETE request', async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({ deleted: true }),
    });

    const result = await api.delete('test-route');
    expect(result).toEqual({ deleted: true });
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining('test-route'),
      expect.objectContaining({ method: 'DELETE' })
    );
  });

  it('should throw MemoriApiRequestRejectedError on 433', async () => {
    const { MemoriApiRequestRejectedError } = await import('../../src/core/errors.js');

    fetchMock.mockResolvedValue({
      ok: false,
      status: 433,
      json: async () => ({ message: 'Rejected' }),
    });

    await expect(api.post('test', {})).rejects.toThrow(MemoriApiRequestRejectedError);
  });
});
