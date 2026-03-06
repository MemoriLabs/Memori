export const PLUGIN_CONFIG = {
  ID: 'openclaw-memori',
  NAME: 'Memori System',
  LOG_PREFIX: '[Memori]',
} as const;

export const RECALL_CONFIG = {
  MIN_PROMPT_LENGTH: 2,
} as const;

export const AUGMENTATION_CONFIG = {
  MAX_CONTEXT_MESSAGES: 5,
} as const;
