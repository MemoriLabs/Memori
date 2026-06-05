import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { Config } from '../../src/core/config.js';

describe('Config', () => {
  const originalEnv = process.env;

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
    delete process.env.MEMORI_ENV;
    delete process.env.MEMORI_DOMAIN;
    delete process.env.MEMORI_API_URL_BASE;
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  it('should load API key from environment', () => {
    process.env.MEMORI_API_KEY = 'env-key';
    const config = new Config();
    expect(config.apiKey).toBe('env-key');
  });

  it('should use env prefix in URL when MEMORI_ENV is set', () => {
    process.env.MEMORI_ENV = 'staging';
    const config = new Config();
    expect(config.testMode).toBe(true);
    expect(config.baseUrl).toBe('https://staging-api.memorilabs.ai');
  });

  it('should use arbitrary env prefix when MEMORI_ENV=qa', () => {
    process.env.MEMORI_ENV = 'qa';
    const config = new Config();
    expect(config.testMode).toBe(true);
    expect(config.baseUrl).toBe('https://qa-api.memorilabs.ai');
  });

  it('should allow overriding base URL via environment', () => {
    process.env.MEMORI_API_URL_BASE = 'https://custom.memori.ai';
    const config = new Config();
    expect(config.baseUrl).toBe('https://custom.memori.ai');
  });

  it('should default to production URL if no env vars set', () => {
    delete process.env.MEMORI_ENV;
    delete process.env.MEMORI_API_URL_BASE;
    const config = new Config();
    expect(config.baseUrl).toBe('https://api.memorilabs.ai');
  });

  describe('tenant domain', () => {
    it('should use MEMORI_DOMAIN for production api subdomain', () => {
      process.env.MEMORI_DOMAIN = 'linkedin.memorilabs.ai';
      const config = new Config();
      expect(config.baseUrl).toBe('https://api.linkedin.memorilabs.ai');
      expect(config.xApiKey).toBe('96a7ea3e-11c2-428c-b9ae-5a168363dc80');
    });

    it('should use MEMORI_DOMAIN with env prefix when MEMORI_ENV=staging', () => {
      process.env.MEMORI_DOMAIN = 'linkedin.memorilabs.ai';
      process.env.MEMORI_ENV = 'staging';
      const config = new Config();
      expect(config.baseUrl).toBe('https://staging-api.linkedin.memorilabs.ai');
      expect(config.xApiKey).toBe('c18b1022-7fe2-42af-ab01-b1f9139184f0');
    });

    it('should use MEMORI_DOMAIN with arbitrary env prefix when MEMORI_ENV=qa', () => {
      process.env.MEMORI_DOMAIN = 'linkedin.memorilabs.ai';
      process.env.MEMORI_ENV = 'qa';
      const config = new Config();
      expect(config.baseUrl).toBe('https://qa-api.linkedin.memorilabs.ai');
      expect(config.xApiKey).toBe('c18b1022-7fe2-42af-ab01-b1f9139184f0');
    });

    it('should prefer MEMORI_DOMAIN over MEMORI_API_URL_BASE', () => {
      process.env.MEMORI_DOMAIN = 'acme.memorilabs.ai';
      process.env.MEMORI_API_URL_BASE = 'https://custom.api.com';
      const config = new Config();
      expect(config.baseUrl).toBe('https://api.acme.memorilabs.ai');
    });
  });
});
