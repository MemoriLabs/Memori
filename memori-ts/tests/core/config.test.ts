import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Config } from '../../src/core/config.js';

const ENTERPRISE_VARS = [
  'MEMORI_ENTERPRISE_PRODUCTION_DOMAIN',
  'MEMORI_ENTERPRISE_STAGING_DOMAIN',
  'MEMORI_API_URL_BASE',
  'MEMORI_TEST_MODE',
];

describe('Config', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
    ENTERPRISE_VARS.forEach((v) => delete process.env[v]);
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
    const config = new Config();
    expect(config.baseUrl).toBe('https://api.memorilabs.ai');
  });

  describe('enterprise domain', () => {
    it('should use enterprise production domain for api subdomain', () => {
      process.env.MEMORI_ENTERPRISE_PRODUCTION_DOMAIN = 'linkedin.memorilabs.ai';
      const config = new Config();
      expect(config.baseUrl).toBe('https://api.linkedin.memorilabs.ai');
      expect(config.xApiKey).toBe('96a7ea3e-11c2-428c-b9ae-5a168363dc80');
    });

    it('should use enterprise staging domain with staging prefix', () => {
      process.env.MEMORI_ENTERPRISE_STAGING_DOMAIN = 'linkedin.memorilabs.ai';
      const config = new Config();
      expect(config.baseUrl).toBe('https://staging-api.linkedin.memorilabs.ai');
      expect(config.xApiKey).toBe('c18b1022-7fe2-42af-ab01-b1f9139184f0');
    });

    it('should prefer enterprise production domain over staging domain', () => {
      process.env.MEMORI_ENTERPRISE_PRODUCTION_DOMAIN = 'acme.memorilabs.ai';
      process.env.MEMORI_ENTERPRISE_STAGING_DOMAIN = 'acme.memorilabs.ai';
      const config = new Config();
      expect(config.baseUrl).toBe('https://api.acme.memorilabs.ai');
    });

    it('should prefer enterprise production domain over MEMORI_API_URL_BASE', () => {
      process.env.MEMORI_ENTERPRISE_PRODUCTION_DOMAIN = 'acme.memorilabs.ai';
      process.env.MEMORI_API_URL_BASE = 'https://custom.api.com';
      const config = new Config();
      expect(config.baseUrl).toBe('https://api.acme.memorilabs.ai');
    });
  });
});
