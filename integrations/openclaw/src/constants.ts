export const PLUGIN_CONFIG = {
  ID: 'openclaw-memori',
  NAME: 'Memori System',
  LOG_PREFIX: '[Memori]',
} as const;

export const RECALL_CONFIG = {
  MIN_PROMPT_LENGTH: 2,
} as const;

export const AUGMENTATION_CONFIG = {
  MAX_CONTEXT_MESSAGES: 20,
} as const;

export const MESSAGE_CONSTANTS = {
  SILENT_REPLY: 'SILENT_REPLY',
  NO_REPLY: 'NO_REPLY',
  SYNTHETIC_RESPONSE: "Okay, I'll remember that for you.",
} as const;

export const ROLE = {
  USER: 'user',
  ASSISTANT: 'assistant',
  TOOL_RESULT: 'toolResult',
  SYSTEM: 'system',
} as const;
