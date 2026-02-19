import { Config } from './config.js';
import {
  MemoriApiClientError,
  MemoriApiRequestRejectedError,
  MemoriApiValidationError,
  QuotaExceededError,
  TimeoutError,
} from './errors.js';

export enum ApiSubdomain {
  DEFAULT = 'api',
  COLLECTOR = 'collector',
}

const PUBLIC_PROD_KEY = '96a7ea3e-11c2-428c-b9ae-5a168363dc80';
const PUBLIC_STAGING_KEY = 'c18b1022-7fe2-42af-ab01-b1f9139184f0';

interface FetchOptions extends RequestInit {
  maxRetries?: number;
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export class Api {
  private readonly config: Config;
  private readonly xApiKey: string;
  private readonly baseUrl: string;

  constructor(config: Config, subdomain: ApiSubdomain = ApiSubdomain.DEFAULT) {
    this.config = config;

    // Switch subdomains based on the target service (e.g. recall vs augmentation).
    // This allows us to use a single base URL config while routing to specific microservices.
    if (subdomain === ApiSubdomain.COLLECTOR) {
      this.baseUrl = this.config.baseUrl
        .replace('://api.', '://collector.')
        .replace('://staging-api.', '://staging-collector.');
    } else {
      this.baseUrl = this.config.baseUrl;
    }

    this.xApiKey = this.config.testMode ? PUBLIC_STAGING_KEY : PUBLIC_PROD_KEY;
  }

  private getHeaders(): HeadersInit {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-Memori-API-Key': this.xApiKey,
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    return headers;
  }

  /**
   * Performs a fetch request with exponential backoff and timeout handling.
   * Retries on network errors, timeouts, and 5xx server errors.
   */
  private async request<T>(
    method: string,
    route: string,
    body?: unknown,
    options: FetchOptions = {}
  ): Promise<T> {
    const url = `${this.baseUrl}/v1/${route}`;
    const { maxRetries = 5, ...fetchInit } = options;

    // Prepare the RequestInit once
    const init: RequestInit = {
      method,
      headers: this.getHeaders(),
      body: body ? JSON.stringify(body) : undefined,
      ...fetchInit,
    };

    let attempt = 0;
    let lastError: Error | null = null;

    while (attempt <= maxRetries) {
      try {
        return await this.executeAttempt<T>(url, init);
      } catch (err: unknown) {
        lastError = err as Error;

        // Determine if we should retry
        const isNetworkError = err instanceof TypeError;
        const isServer5xx =
          err instanceof MemoriApiClientError && err.statusCode >= 500 && err.statusCode <= 599;
        const isTimeout = err instanceof TimeoutError;

        if ((isNetworkError || isServer5xx || isTimeout) && attempt < maxRetries) {
          const backoff = Math.pow(2, attempt) * 1000;
          await delay(backoff);
          attempt++;
          continue;
        }

        throw lastError;
      }
    }

    throw lastError || new Error('Unknown network error');
  }

  private async executeAttempt<T>(url: string, init: RequestInit): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      controller.abort();
    }, this.config.timeout);

    try {
      const response = await fetch(url, { ...init, signal: controller.signal });
      clearTimeout(timeoutId);

      if (response.ok) {
        if (response.status === 204) return {} as T;

        return (await response.json()) as T;
      }

      await this.throwOnApiError(response);
      throw new Error('Unreachable');
    } catch (err: unknown) {
      clearTimeout(timeoutId);

      if ((err as Error).name === 'AbortError') {
        throw new TimeoutError(this.config.timeout);
      }
      throw err;
    }
  }

  private async throwOnApiError(response: Response): Promise<void> {
    // Try to parse the error details safely
    let errorData: { message?: string; detail?: string } = {};

    try {
      errorData = (await response.json()) as typeof errorData;
    } catch {
      // ignore parsing error
    }

    const message = errorData.message || errorData.detail;
    const status = response.status;

    // Map status codes to errors
    if (status === 429) throw new QuotaExceededError(message);
    if (status === 422) {
      throw new MemoriApiValidationError(
        status,
        message || 'Memori API rejected the request (422 validation error).'
      );
    }
    if (status === 433) {
      throw new MemoriApiRequestRejectedError(status, message || 'The request was rejected (433).');
    }
    if (status >= 500) throw new MemoriApiClientError(status, message);
    if (status >= 400) throw new MemoriApiClientError(status, message);

    throw new MemoriApiClientError(status, `Unknown error ${status}`);
  }

  public get<T>(route: string): Promise<T> {
    return this.request<T>('GET', route);
  }

  public post<T>(route: string, body?: unknown): Promise<T> {
    return this.request<T>('POST', route, body);
  }

  public patch<T>(route: string, body?: unknown): Promise<T> {
    return this.request<T>('PATCH', route, body);
  }

  public delete<T>(route: string): Promise<T> {
    return this.request<T>('DELETE', route);
  }
}
