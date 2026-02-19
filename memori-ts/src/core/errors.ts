export class MemoriError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'MemoriError';
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

export class QuotaExceededError extends MemoriError {
  constructor(message?: string) {
    super(
      message ||
        'Your IP address is over quota; register for an API key now: https://app.memorilabs.ai/signup'
    );
    this.name = 'QuotaExceededError';
  }
}

export class MemoriApiClientError extends MemoriError {
  public readonly statusCode: number;
  public readonly details?: unknown;

  constructor(statusCode: number, message?: string, details?: unknown) {
    super(message || `Memori API request failed with status ${statusCode}`);
    this.name = 'MemoriApiClientError';
    this.statusCode = statusCode;
    this.details = details;
  }
}

export class MemoriApiValidationError extends MemoriApiClientError {
  constructor(statusCode: number, message: string, details?: unknown) {
    super(statusCode, message, details);
    this.name = 'MemoriApiValidationError';
  }
}

export class MemoriApiRequestRejectedError extends MemoriApiClientError {
  constructor(statusCode: number, message: string, details?: unknown) {
    super(statusCode, message, details);
    this.name = 'MemoriApiRequestRejectedError';
  }
}

export class MissingMemoriApiKeyError extends MemoriError {
  constructor(envVar = 'MEMORI_API_KEY') {
    super(
      `A ${envVar} is required to use the Memori hosted API. Sign up at https://app.memorilabs.ai/signup`
    );
    this.name = 'MissingMemoriApiKeyError';
  }
}

export class TimeoutError extends MemoriError {
  constructor(timeout: number) {
    super(`Request timed out after ${timeout}ms`);
    this.name = 'TimeoutError';
  }
}
