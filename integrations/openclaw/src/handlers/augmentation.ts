import { IntegrationRequest, IntegrationMetadata } from '@memorilabs/memori/integrations';
import { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from '../types.js';
import { extractContext, MemoriLogger, initializeMemoriClient } from '../utils/index.js';
import { cleanText, isSystemMessage } from '../sanitizer.js';
import { SDK_VERSION } from '../version.js';
import { AUGMENTATION_CONFIG } from '../constants.js';

function extractLLMMetadata(event: OpenClawEvent): IntegrationMetadata {
  const messages = event.messages || [];
  const lastAssistant = messages.findLast((m) => m.role === 'assistant');

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

  if (!event.success || !event.messages || event.messages.length === 0) {
    logger.info('No messages or unsuccessful event. Skipping augmentation.');
    logger.endSection('AUGMENTATION HOOK END');
    return;
  }

  const recentMessages = event.messages.slice(-AUGMENTATION_CONFIG.MAX_CONTEXT_MESSAGES);
  const formattedMessages: Array<{ role: string; content: string }> = [];

  for (const msg of recentMessages) {
    const role = msg.role;
    if (role !== 'user' && role !== 'assistant') continue;

    const cleanedContent = cleanText(msg.content);

    if (!cleanedContent) continue;

    let finalContent = cleanedContent;
    if (role === 'assistant') {
      finalContent = finalContent.replace(/^\[\[.*?\]\]\s*/, '');
    }

    formattedMessages.push({
      role: role as string,
      content: finalContent,
    });
  }

  const lastUserMsg = formattedMessages.findLast((m) => m.role === 'user');
  let lastAiMsg = formattedMessages.findLast((m) => m.role === 'assistant');

  if (!lastUserMsg || !lastAiMsg) {
    logger.info('Missing user or assistant message. Skipping.');
    logger.endSection('AUGMENTATION HOOK END');
    return;
  }

  // Check if the user message is a system message that should be ignored
  if (isSystemMessage(lastUserMsg.content)) {
    logger.info('User message is a system message. Skipping augmentation.');
    logger.endSection('AUGMENTATION HOOK END');
    return;
  }

  if (lastAiMsg.content === 'NO_REPLY' || lastAiMsg.content === 'SILENT_REPLY') {
    logger.info(
      'Assistant used tool-based messaging (NO_REPLY). Using synthetic response for augmentation.'
    );
    lastAiMsg = {
      role: 'assistant',
      content: "Okay, I'll remember that for you.",
    };
  }

  const context = extractContext(event, ctx, config.entityId);
  const memoriClient = initializeMemoriClient(config.apiKey, context);

  try {
    logger.info(`Capturing conversation turn...`);
    const payload: IntegrationRequest = {
      userMessage: lastUserMsg.content,
      agentResponse: lastAiMsg.content,
      metadata: extractLLMMetadata(event),
    };

    await memoriClient.augmentation(payload);
    logger.info('Augmentation successful!');
  } catch (err) {
    logger.error(`Augmentation failed: ${err instanceof Error ? err.message : String(err)}`);
  }

  logger.endSection('AUGMENTATION HOOK END');
}
