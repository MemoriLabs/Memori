import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { getOrCreateSession } from '../manager.js';
import { cleanText, isSystemMessage } from '../sanitizer.js';
import { OpenClawEvent, OpenClawContext } from '../types.js';
import { extractContext } from '../utils/context.js';
import { MemoriLogger } from '../utils/logger.js';
import { initializeMemoriClient } from '../utils/memori-client.js';
import { IntegrationRequest, IntegrationMetadata } from '@memorilabs/memori/integrations';
import { SDK_VERSION } from '../version.js';

/**
 * Extracts the last user and assistant messages from the conversation
 */
function getLastMessages(event: OpenClawEvent) {
  const messages = event.messages || [];
  return {
    user: messages.filter((m) => m.role === 'user').at(-1),
    assistant: messages.filter((m) => m.role === 'assistant').at(-1),
  };
}

function extractLLMMetadata(event: OpenClawEvent): IntegrationMetadata {
  const messages = event.messages || [];
  const lastAssistant = messages.filter((m) => m.role === 'assistant').at(-1);

  return {
    provider: lastAssistant?.provider as string || null,
    model: lastAssistant?.model as string || null,
    sdkVersion: null,
    integrationSdkVersion: SDK_VERSION,
    platform: 'openclaw',
  }
}

/**
 * Handles memory capture after agent completion.
 * Sends the conversation turn to Memori for extraction and storage.
 *
 * @param event - OpenClaw event containing conversation messages
 * @param ctx - OpenClaw context with session information
 * @param api - Plugin API for logging
 * @param apiKey - Memori API key
 * @param configuredEntityId - Optional hardcoded entity ID
 */
export async function handleAugmentation(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  api: OpenClawPluginApi,
  apiKey: string,
  configuredEntityId: string | undefined
) {
  const logger = new MemoriLogger(api);
  logger.section('AUGMENTATION HOOK START');

  const context = extractContext(event, ctx, configuredEntityId);
  const { user: lastUserMsg, assistant: lastAssistantMsg } = getLastMessages(event);

  const userText = cleanText(lastUserMsg?.content);
  const aiText = cleanText(lastAssistantMsg?.content);

  if (!userText || !aiText || isSystemMessage(userText) || isSystemMessage(aiText)) {
    logger.info('Empty or system message detected. Skipping augmentation.');
    return;
  }

  const session = getOrCreateSession(context.entityId, apiKey);
  const memoriClient = initializeMemoriClient(session, context);

  try {
    logger.info(`Capturing conversation turn...`);
    const payload: IntegrationRequest = {
      userMessage: userText,
      agentResponse: aiText,
      metadata: extractLLMMetadata(event)
    };

    await memoriClient.augmentation(payload);
    logger.info('Augmentation successful!');
  } catch (err) {
    logger.error(`Augmentation failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  logger.endSection('Augmentation HOOK END');
}
