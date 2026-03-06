export const PLUGIN_CONFIG = {
  /** Plugin identifier */
  ID: 'openclaw-memori',

  /** Display name */
  NAME: 'Memori System',

  /** Log prefix for all plugin messages */
  LOG_PREFIX: '[Memori]',
} as const;

export const RECALL_CONFIG = {
  /** Minimum prompt length to trigger recall */
  MIN_PROMPT_LENGTH: 2,
} as const;

export const AUGMENTATION_CONFIG = {
  MAX_CONTEXT_MESSAGES: 10,
} as const;
