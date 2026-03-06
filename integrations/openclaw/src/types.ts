export interface MemoriPluginConfig {
  apiKey: string;
  entityId: string;
}

export interface OpenClawMessageBlock {
  type?: string;
  text?: string;
  thinking?: string;
  [key: string]: unknown;
}

export interface OpenClawMessage {
  role: 'user' | 'assistant' | 'system';
  content: string | OpenClawMessageBlock[];
  timestamp?: number;
  [key: string]: unknown;
}

export interface OpenClawEvent {
  prompt?: string;
  messages?: OpenClawMessage[];
  completion?: string;
  success?: boolean;
  error?: string;
  durationMs?: number;
  userId?: string;
  sessionId?: string;
  messageProvider?: string;
}

export interface OpenClawContext {
  agentId?: string;
  sessionKey?: string;
  sessionId?: string;
  workspaceDir?: string;
  messageProvider?: string;
}
