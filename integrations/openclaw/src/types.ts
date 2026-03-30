export interface MemoriPluginConfig {
  apiKey: string;
  entityId: string;
}

export interface OpenClawMessageBlock {
  type?: string;
  text?: string;
  thinking?: string;
  name?: string;
  id?: string;
  arguments?: unknown;
  [key: string]: unknown;
}

export interface OpenClawMessage {
  role: string;
  content: string | OpenClawMessageBlock[];
  timestamp?: number;
  toolCallId?: string;
  provider?: string;
  model?: string;
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

export interface ExtractedToolCall {
  name: string;
  args: Record<string, unknown>;
  result: unknown;
}

export interface ParsedTurn {
  userMessage: { role: string; content: string } | null;
  assistantMessage: { role: string; content: string } | null;
  tools: ExtractedToolCall[];
}
