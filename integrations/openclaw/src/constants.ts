/**
 * Plugin configuration constants
 */
export const PLUGIN_CONFIG = {
  /** Plugin identifier */
  ID: 'openclaw-memori',

  /** Display name */
  NAME: 'Memori System',

  /** Log prefix for all plugin messages */
  LOG_PREFIX: '[Memori]',
} as const;

/**
 * Session management constants
 */
export const SESSION_CONFIG = {
  /** Session cache TTL in milliseconds (24 hours) */
  TTL_MS: 1000 * 60 * 60 * 24,

  /** Garbage collection interval in milliseconds (10 minutes) */
  GC_INTERVAL_MS: 1000 * 60 * 10,

  /** Minimum interval between API calls (1 second) */
  MIN_API_INTERVAL_MS: 1000,
} as const;

/**
 * Recall configuration
 */
export const RECALL_CONFIG = {
  /** Minimum prompt length to trigger recall */
  MIN_PROMPT_LENGTH: 2,
} as const;

/**
 * Default fallback values
 */
export const DEFAULTS = {
  USER_ID: 'default-user',
  SESSION_ID: 'default-session',
  PROVIDER: 'openclaw',
} as const;
