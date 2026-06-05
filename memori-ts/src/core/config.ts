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
   * Whether the SDK is running in staging mode.
   * Defaults to `true` if `MEMORI_ENV` is set to 'staging'.
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
    const envPrefix = getEnv('MEMORI_ENV')?.trim() ?? '';
    this.testMode = envPrefix !== '';

    const domain = getEnv('MEMORI_DOMAIN')?.trim();
    if (domain) {
      this.baseUrl = envPrefix ? `https://${envPrefix}-api.${domain}` : `https://api.${domain}`;
      this.xApiKey = envPrefix ? PUBLIC_STAGING_KEY : PUBLIC_PROD_KEY;
    } else {
      const envUrl = getEnv('MEMORI_API_URL_BASE');
      if (envUrl) {
        this.baseUrl = envUrl;
        this.xApiKey = PUBLIC_STAGING_KEY;
      } else {
        this.baseUrl = envPrefix
          ? `https://${envPrefix}-api.memorilabs.ai`
          : 'https://api.memorilabs.ai';
        this.xApiKey = envPrefix ? PUBLIC_STAGING_KEY : PUBLIC_PROD_KEY;
      }
    }

    this.apiKey = getEnv('MEMORI_API_KEY') ?? null;
    this.sessionId = randomUUID();
    this.recallRelevanceThreshold = 0.1;
    this.timeout = 30000;
  }
}
