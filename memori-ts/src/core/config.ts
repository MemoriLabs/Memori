import { randomUUID } from 'node:crypto';

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

  constructor() {
    // 1. Environment and Base URL Logic
    this.testMode = getEnv('MEMORI_TEST_MODE') === '1';

    const envUrl = getEnv('MEMORI_API_URL_BASE');
    if (envUrl) {
      this.baseUrl = envUrl;
    } else {
      this.baseUrl = this.testMode
        ? 'https://staging-api.memorilabs.ai'
        : 'https://api.memorilabs.ai';
    }

    // 2. Authentication
    this.apiKey = getEnv('MEMORI_API_KEY') ?? null;

    // 3. Session and Defaults
    this.sessionId = randomUUID();
    this.recallRelevanceThreshold = 0.1; // Default parity with Python
    this.timeout = 5000; // 5 seconds
  }
}
