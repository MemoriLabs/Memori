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
 * Recall configuration
 */
export const RECALL_CONFIG = {
  /** Minimum prompt length to trigger recall */
  MIN_PROMPT_LENGTH: 2,
} as const;
