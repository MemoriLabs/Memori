import { createRequire } from 'node:module';
import { randomUUID } from 'node:crypto';
import type { StorageManager } from '../storage/manager.js';

const PUBLIC_PROD_KEY = '96a7ea3e-11c2-428c-b9ae-5a168363dc80';
const PUBLIC_STAGING_KEY = 'c18b1022-7fe2-42af-ab01-b1f9139184f0';

/**
 * Utility to safely retrieve environment variables across Node.js and other runtimes.
 */
function getEnv(key: string): string | undefined {
  if (typeof process !== 'undefined') {
    return process.env[key];
  }
  return undefined;
}

export class Config {
  /**
   * The API Key used for authentication.
   * Defaults to `MEMORI_API_KEY` environment variable.
   */
  public apiKey: string | null;

  /**
   * The base URL for the Memori API.
   * Automatically switches between production and staging based on `testMode`.
   */
  public baseUrl: string;

  /**
   * The X-Memori-API-Key header value derived from the active environment.
   */
  public xApiKey: string;

  /**
   * Whether the SDK is running in test/staging mode.
   * Defaults to `true` if `MEMORI_TEST_MODE` is set to '1'.
   */
  public testMode: boolean;

  /**
   * The unique identifier for the end-user associated with the current memories.
   */
  public entityId?: string;

  /**
   * The unique identifier for the specific process or workflow.
   */
  public processId?: string;

  /**
   * The current conversation session ID.
   * Included in all requests to track conversation history.
   */
  public sessionId: string;

  /**
   * The minimum relevance score (0.0 to 1.0) required for a memory to be included in the context.
   * Defaults to 0.1.
   */
  public recallRelevanceThreshold: number;

  /**
   * Request timeout in milliseconds.
   * Defaults to 5000ms (5 seconds).
   */
  public timeout: number;

  /**
   * The active storage manager handling local database operations.
   * Only populated if a database connection is provided to Memori.
   */
  public storage?: StorageManager;

  constructor() {
    this.testMode = getEnv('MEMORI_TEST_MODE') === '1';

    // Delegate URL and key resolution to Rust so the enterprise domain env vars
    // have a single source of truth. Falls back to reading env vars directly
    // when native bindings are unavailable (e.g. unit tests).
    try {
      const require = createRequire(import.meta.url);
      const native = require('../native/index.js') as {
        resolveApiBaseUrl: (subdomain: string) => string;
        resolveXApiKey: () => string;
      };
      this.baseUrl = native.resolveApiBaseUrl('api');
      this.xApiKey = native.resolveXApiKey();
    } catch {
      this.baseUrl = this._resolveBaseUrlFallback();
      this.xApiKey = this._resolveXApiKeyFallback();
    }

    this.apiKey = getEnv('MEMORI_API_KEY') ?? null;
    this.sessionId = randomUUID();
    this.recallRelevanceThreshold = 0.1;
    this.timeout = 30000;
  }

  // Mirrors Rust resolve_base_url — used only when native bindings are unavailable.
  private _resolveBaseUrlFallback(): string {
    const enterpriseProd = getEnv('MEMORI_ENTERPRISE_PRODUCTION_DOMAIN')?.trim();
    if (enterpriseProd) return `https://api.${enterpriseProd}`;

    const enterpriseStaging = getEnv('MEMORI_ENTERPRISE_STAGING_DOMAIN')?.trim();
    if (enterpriseStaging) return `https://staging-api.${enterpriseStaging}`;

    const envUrl = getEnv('MEMORI_API_URL_BASE');
    if (envUrl) return envUrl;

    return this.testMode ? 'https://staging-api.memorilabs.ai' : 'https://api.memorilabs.ai';
  }

  // Mirrors Rust resolve_x_api_key — used only when native bindings are unavailable.
  private _resolveXApiKeyFallback(): string {
    const enterpriseProd = getEnv('MEMORI_ENTERPRISE_PRODUCTION_DOMAIN')?.trim();
    if (enterpriseProd) return PUBLIC_PROD_KEY;

    const usesStaging =
      !!getEnv('MEMORI_ENTERPRISE_STAGING_DOMAIN')?.trim() ||
      !!getEnv('MEMORI_API_URL_BASE') ||
      this.testMode;

    return usesStaging ? PUBLIC_STAGING_KEY : PUBLIC_PROD_KEY;
  }
}
