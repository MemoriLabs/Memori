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
   * The base URL for the Memori API (always the `api` subdomain).
   * Use `resolveUrl(subdomain)` to get the URL for a specific subdomain.
   */
  public baseUrl: string;

  /**
   * The X-Memori-API-Key header value derived from the active environment.
   */
  public xApiKey: string;

  /**
   * Whether the SDK is running in a non-production environment.
   * True when `MEMORI_ENV` is set to any non-empty value.
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

  private readonly _envPrefix: string;
  private readonly _domain: string | undefined;
  private readonly _apiUrlBase: string | undefined;

  constructor() {
    this._envPrefix = getEnv('MEMORI_ENV')?.trim() ?? '';
    this._domain = getEnv('MEMORI_DOMAIN')?.trim() || undefined;
    this._apiUrlBase = getEnv('MEMORI_API_URL_BASE') || undefined;

    this.testMode = this._envPrefix !== '';
    this.baseUrl = this.resolveUrl('api');
    this.xApiKey =
      this._domain || !this._apiUrlBase
        ? this._envPrefix
          ? PUBLIC_STAGING_KEY
          : PUBLIC_PROD_KEY
        : PUBLIC_STAGING_KEY;

    this.apiKey = getEnv('MEMORI_API_KEY') ?? null;
    this.sessionId = randomUUID();
    this.recallRelevanceThreshold = 0.1;
    this.timeout = 30000;
  }

  /**
   * Builds the full base URL for the given subdomain (e.g. `'api'`, `'collector'`).
   * Mirrors the Python `_resolve_api_config` logic.
   */
  public resolveUrl(subdomain: string): string {
    if (this._domain) {
      return this._envPrefix
        ? `https://${this._envPrefix}-${subdomain}.${this._domain}`
        : `https://${subdomain}.${this._domain}`;
    }
    if (this._apiUrlBase) return this._apiUrlBase;
    return this._envPrefix
      ? `https://${this._envPrefix}-${subdomain}.memorilabs.ai`
      : `https://${subdomain}.memorilabs.ai`;
  }
}
