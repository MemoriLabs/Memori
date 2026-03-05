import { IntegrationRequest, IntegrationMetadata } from '@memorilabs/memori/integrations';
import { cleanText, isSystemMessage } from '../sanitizer.js';
import { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from '../types.js';
import { extractContext, MemoriLogger, initializeMemoriClient } from '../utils/index.js';
import { SDK_VERSION } from '../version.js';

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
    provider: (lastAssistant?.provider as string) || null,
    model: (lastAssistant?.model as string) || null,
    sdkVersion: null,
    integrationSdkVersion: SDK_VERSION,
    platform: 'openclaw',
  };
}

export async function handleAugmentation(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  config: MemoriPluginConfig,
  logger: MemoriLogger
): Promise<void> {
  logger.section('AUGMENTATION HOOK START');

  const context = extractContext(event, ctx, config.entityId);
  const { user: lastUserMsg, assistant: lastAssistantMsg } = getLastMessages(event);

  const userText = cleanText(lastUserMsg?.content);
  const aiText = cleanText(lastAssistantMsg?.content);

  if (!userText || !aiText || isSystemMessage(userText) || isSystemMessage(aiText)) {
    logger.info('Empty or system message detected. Skipping augmentation.');
    return;
  }

  const memoriClient = initializeMemoriClient(config.apiKey, context);

  try {
    logger.info(`Capturing conversation turn...`);
    const payload: IntegrationRequest = {
      userMessage: userText,
      agentResponse: aiText,
      metadata: extractLLMMetadata(event),
    };

    await memoriClient.augmentation(payload);
    logger.info('Augmentation successful!');
  } catch (err) {
    logger.error(`Augmentation failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  logger.endSection('AUGMENTATION HOOK END');
}
