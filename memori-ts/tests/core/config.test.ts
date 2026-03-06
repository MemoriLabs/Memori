import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Config } from '../../src/core/config.js';

describe('Config', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should load API key from environment', () => {
    process.env.MEMORI_API_KEY = 'env-key';
    const config = new Config();
    expect(config.apiKey).toBe('env-key');
  });

  it('should use staging URL if test mode is enabled via env', () => {
    process.env.MEMORI_TEST_MODE = '1';
    const config = new Config();
    expect(config.testMode).toBe(true);
    expect(config.baseUrl).toContain('staging-api');
  });

  it('should allow overriding base URL via environment', () => {
    process.env.MEMORI_API_URL_BASE = 'https://custom.memori.ai';
    const config = new Config();
    expect(config.baseUrl).toBe('https://custom.memori.ai');
  });

  it('should default to production URL if no env vars set', () => {
    delete process.env.MEMORI_TEST_MODE;
    delete process.env.MEMORI_API_URL_BASE;
    const config = new Config();
    expect(config.baseUrl).toBe('https://api.memorilabs.ai');
  });
});
