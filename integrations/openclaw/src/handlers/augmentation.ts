import { IntegrationRequest, IntegrationMetadata } from '@memorilabs/memori/integrations';
import {
  OpenClawEvent,
  OpenClawContext,
  MemoriPluginConfig,
  ExtractedToolCall,
  OpenClawMessage,
  ParsedTurn,
} from '../types.js';
import { extractContext, MemoriLogger, initializeMemoriClient } from '../utils/index.js';
import { cleanText, isSystemMessage } from '../sanitizer.js';
import { AUGMENTATION_CONFIG, MESSAGE_CONSTANTS, ROLE } from '../constants.js';
import { SDK_VERSION } from '../version.js';

/**
 * Extracts metadata about the LLM provider and model used during the turn.
 */
function extractLLMMetadata(event: OpenClawEvent): IntegrationMetadata {
  const messages = event.messages || [];
  const lastAssistant = messages.findLast((m) => m.role === ROLE.ASSISTANT);

  return {
    provider: (lastAssistant?.provider as string) || null,
    model: (lastAssistant?.model as string) || null,
    sdkVersion: null,
    integrationSdkVersion: SDK_VERSION,
    platform: 'openclaw',
  };
}

/**
 * Extracts all tool results from messages and maps them by tool call ID.
 */
function extractToolResults(messages: OpenClawMessage[]): Map<string, unknown> {
  const resultsMap = new Map<string, unknown>();

  for (const msg of messages) {
    if (msg.role === ROLE.TOOL_RESULT && msg.toolCallId) {
      resultsMap.set(msg.toolCallId, cleanText(msg.content));
    }
  }

  return resultsMap;
}

/**
 * Parses tool arguments from various formats into a structured object.
 */
function parseToolArguments(rawArgs: unknown): Record<string, unknown> {
  if (typeof rawArgs === 'string') {
    try {
      return JSON.parse(rawArgs) as Record<string, unknown>;
    } catch {
      return {};
    }
  }

  if (typeof rawArgs === 'object' && rawArgs !== null) {
    return rawArgs as Record<string, unknown>;
  }

  return {};
}

/**
 * Extracts tool calls from an assistant message.
 * Iterates backward through content blocks to match original extraction order.
 */
function extractToolCalls(
  msg: OpenClawMessage,
  toolResults: Map<string, unknown>
): ExtractedToolCall[] {
  if (!Array.isArray(msg.content)) {
    return [];
  }

  const tools: ExtractedToolCall[] = [];

  for (let i = msg.content.length - 1; i >= 0; i--) {
    const block = msg.content[i];

    if (block.type === 'toolCall' || block.type === 'tool_use') {
      const rawArgs = block.arguments ?? block.input;

      // Unshift to maintain chronological order within the message
      tools.unshift({
        name: block.name as string,
        args: parseToolArguments(rawArgs),
        result: toolResults.get(block.id as string) ?? null,
      });
    }
  }

  return tools;
}

/**
 * Parses a user message, extracting the cleaned content.
 */
function parseUserMessage(msg: OpenClawMessage): { role: string; content: string } | null {
  const cleanedContent = cleanText(msg.content);
  return cleanedContent ? { role: msg.role, content: cleanedContent } : null;
}

/**
 * Parses the most recent conversation turn from messages.
 * Walks backward to find the last user message and assistant response.
 * Collects all tool calls from assistant messages in the turn.
 */
function parseTurnFromMessages(messages: OpenClawMessage[]): ParsedTurn {
  const tools: ExtractedToolCall[] = [];
  const toolResults = extractToolResults(messages);

  let userMessage = null;
  let assistantMessage = null;

  // Walk backwards to find the last complete turn
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i];

    if (msg.role === ROLE.ASSISTANT) {
      // Extract tool calls from ALL assistant messages in the turn
      const extractedTools = extractToolCalls(msg, toolResults);
      if (extractedTools.length > 0) {
        // Prepend to maintain chronological order
        tools.unshift(...extractedTools);
      }

      // Capture the text response from the FIRST (most recent) assistant message
      if (!assistantMessage) {
        const cleanedContent = cleanText(msg.content);
        if (cleanedContent) {
          assistantMessage = {
            role: msg.role,
            content: cleanedContent.replace(/^\[\[.*?\]\]\s*/, ''),
          };
        } else if (extractedTools.length > 0) {
          assistantMessage = { role: msg.role, content: MESSAGE_CONSTANTS.SILENT_REPLY };
        }
      }
    } else if (msg.role === ROLE.USER) {
      userMessage = parseUserMessage(msg);
      break; // Found the user message that started this turn
    }
  }

  return { userMessage, assistantMessage, tools };
}

/**
 * Builds the augmentation payload to send to Memori backend.
 */
function buildAugmentationPayload(
  userMessage: string,
  agentResponse: string,
  tools: ExtractedToolCall[],
  event: OpenClawEvent
): IntegrationRequest {
  const payload: IntegrationRequest = {
    userMessage,
    agentResponse,
    metadata: extractLLMMetadata(event),
  };

  if (tools.length > 0) {
    payload.trace = { tools };
  }

  return payload;
}

/**
 * Helper to skip augmentation and log the reason.
 */
function skipAugmentation(logger: MemoriLogger, reason: string): void {
  logger.info(reason);
  logger.endSection('AUGMENTATION HOOK END');
}

/**
 * Main handler for augmentation hook.
 * Extracts the latest conversation turn and sends it to Memori backend.
 */
export async function handleAugmentation(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  config: MemoriPluginConfig,
  logger: MemoriLogger
): Promise<void> {
  logger.section('AUGMENTATION HOOK START');

  if (!event.success || !event.messages || event.messages.length < 2) {
    skipAugmentation(logger, 'No messages or unsuccessful event. Skipping augmentation.');
    return;
  }

  try {
    const recentMessages = event.messages.slice(-AUGMENTATION_CONFIG.MAX_CONTEXT_MESSAGES);
    const turn = parseTurnFromMessages(recentMessages);

    if (!turn.userMessage || !turn.assistantMessage) {
      skipAugmentation(logger, 'Missing user or assistant message. Skipping.');
      return;
    }

    if (isSystemMessage(turn.userMessage.content)) {
      skipAugmentation(logger, 'User message is a system message. Skipping augmentation.');
      return;
    }

    // Resolve synthetic responses for pure-tool executions
    if (
      turn.assistantMessage.content === MESSAGE_CONSTANTS.SILENT_REPLY ||
      turn.assistantMessage.content === MESSAGE_CONSTANTS.NO_REPLY
    ) {
      logger.info('Assistant used tool-based messaging. Using synthetic response.');
      turn.assistantMessage.content = MESSAGE_CONSTANTS.SYNTHETIC_RESPONSE;
    }

    const payload = buildAugmentationPayload(
      turn.userMessage.content,
      turn.assistantMessage.content,
      turn.tools,
      event
    );

    const context = extractContext(event, ctx, config.entityId);
    const memoriClient = initializeMemoriClient(config.apiKey, context);

    await memoriClient.augmentation(payload);

    logger.info('Augmentation successful!');
  } catch (err) {
    logger.error(`Augmentation failed: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    logger.endSection('AUGMENTATION HOOK END');
  }
}
