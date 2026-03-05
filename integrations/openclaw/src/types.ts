/**
 * Configuration options passed to the OpenClaw Memori plugin
 */
export interface MemoriPluginConfig {
  /** Memori API Key */
  apiKey: string;

  /** EntityID used for Recall and Augemtation*/
  entityId: string;
}

/**
 * Represents a single block in OpenClaw's multi-modal message format.
 * Messages can contain text, thinking blocks, images, and other content types.
 */
export interface OpenClawMessageBlock {
  /** Block type (e.g., 'text', 'thinking', 'image') */
  type?: string;

  /** Text content for text blocks */
  text?: string;

  /** Thinking content for internal reasoning blocks */
  thinking?: string;

  /** Additional properties may be present depending on block type */
  [key: string]: unknown;
}

/**
 * Represents a message in an OpenClaw conversation.
 * Can be from user, assistant, or system.
 */
export interface OpenClawMessage {
  /** Message role - who sent this message */
  role: 'user' | 'assistant' | 'system';

  /** Message content - can be plain string or multi-modal blocks */
  content: string | OpenClawMessageBlock[];

  /** Unix timestamp of when the message was created */
  timestamp?: number;

  /** Additional metadata may be present */
  [key: string]: unknown;
}

/**
 * Event object passed to plugin hooks.
 * Contains information about the current interaction and conversation state.
 */
export interface OpenClawEvent {
  /** Current user prompt being processed */
  prompt?: string;

  /** Full conversation history */
  messages?: OpenClawMessage[];

  /** Agent's completion text (for agent_end hook) */
  completion?: string;

  /** Whether the agent execution succeeded */
  success?: boolean;

  /** Error message if execution failed */
  error?: string;

  /** Duration of agent execution in milliseconds */
  durationMs?: number;

  /** User/entity identifier */
  userId?: string;

  /** Session identifier */
  sessionId?: string;

  /** Messaging provider (e.g., 'discord', 'telegram', 'webchat') */
  messageProvider?: string;
}

/**
 * Context object passed to plugin hooks.
 * Contains session and workspace information.
 */
export interface OpenClawContext {
  /** Agent identifier */
  agentId?: string;

  /** Session key for tracking conversation state */
  sessionKey?: string;

  /** Session ID (alternative to sessionKey) */
  sessionId?: string;

  /** Path to agent's workspace directory */
  workspaceDir?: string;

  /** Messaging provider (e.g., 'discord', 'telegram', 'webchat') */
  messageProvider?: string;
}
