export { UnsupportedLLMProviderError } from '@memorilabs/axon';

/** Base class for all Memori SDK errors. */
export class MemoriError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MemoriError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/** Thrown when the Memori Cloud IP rate limit or account quota is exceeded. */
export class QuotaExceededError extends MemoriError {
  constructor(message?: string) {
    super(
      message ||
        'Your IP address is over quota; register for an API key now: https://app.memorilabs.ai/signup'
    );
    this.name = 'QuotaExceededError';
  }
}

/** Thrown when the Memori Cloud API returns a 4xx or 5xx status code. */
export class MemoriApiClientError extends MemoriError {
  /** The HTTP status code returned by the API. */
  public readonly statusCode: number;
  /** Optional details provided by the API response. */
  public readonly details?: unknown;

  constructor(statusCode: number, message?: string, details?: unknown) {
    super(message || `Memori API request failed with status ${statusCode}`);
    this.name = 'MemoriApiClientError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

/** Thrown when the Memori Cloud API rejects a request due to validation errors (422). */
export class MemoriApiValidationError extends MemoriApiClientError {
  constructor(statusCode: number, message: string, details?: unknown) {
    super(statusCode, message, details);
    this.name = 'MemoriApiValidationError';
  }
}

/** Thrown when the Memori Cloud API explicitly rejects a request (433). */
export class MemoriApiRequestRejectedError extends MemoriApiClientError {
  constructor(statusCode: number, message: string, details?: unknown) {
    super(statusCode, message, details);
    this.name = 'MemoriApiRequestRejectedError';
  }
}

/** Thrown when a request requires an API key but none was found in the environment or config. */
export class MissingMemoriApiKeyError extends MemoriError {
  constructor(envVar = 'MEMORI_API_KEY') {
    super(
      `A ${envVar} is required to use the Memori cloud API. Sign up at https://app.memorilabs.ai/signup`
    );
    this.name = 'MissingMemoriApiKeyError';
  }
}

/** Thrown when a network request to the Memori API exceeds the configured timeout duration. */
export class TimeoutError extends MemoriError {
  constructor(timeout: number) {
    super(`Request timed out after ${timeout}ms`);
    this.name = 'TimeoutError';
  }
}
