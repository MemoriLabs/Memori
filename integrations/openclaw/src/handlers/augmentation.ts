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

  if (!event.success || !event.messages || event.messages.length < 2) {
    logger.info('No messages or unsuccessful event. Skipping augmentation.');
    logger.endSection('AUGMENTATION HOOK END');
    return;
  }

  try {
    const recentMessages = event.messages.slice(-AUGMENTATION_CONFIG.MAX_CONTEXT_MESSAGES);

    let lastUserMsg: { role: string; content: string } | undefined;
    let lastAiMsg: { role: string; content: string } | undefined;

    for (let i = recentMessages.length - 1; i >= 0; i--) {
      const msg = recentMessages[i];
      const role = msg.role;

      if (role !== 'user' && role !== 'assistant') continue;

      const cleanedContent = cleanText(msg.content);
      if (!cleanedContent) continue;

      let finalContent = cleanedContent;
      if (role === 'assistant') {
        finalContent = finalContent.replace(/^\[\[.*?\]\]\s*/, '');
      }

      if (role === 'assistant' && !lastAiMsg) {
        lastAiMsg = { role, content: finalContent };
      }

      if (role === 'user' && !lastUserMsg) {
        lastUserMsg = { role, content: finalContent };
      }

      if (lastUserMsg && lastAiMsg) break;
    }

    if (!lastUserMsg || !lastAiMsg) {
      logger.info('Missing user or assistant message. Skipping.');
      logger.endSection('AUGMENTATION HOOK END');
      return;
    }

    if (isSystemMessage(lastUserMsg.content)) {
      logger.info('User message is a system message. Skipping augmentation.');
      logger.endSection('AUGMENTATION HOOK END');
      return;
    }

    if (lastAiMsg.content === 'NO_REPLY' || lastAiMsg.content === 'SILENT_REPLY') {
      logger.info('Assistant used tool-based messaging. Using synthetic response.');
      lastAiMsg = {
        role: 'assistant',
        content: "Okay, I'll remember that for you.",
      };
    }

    const context = extractContext(event, ctx, config.entityId);
    const memoriClient = initializeMemoriClient(config.apiKey, context);

    logger.info('Capturing conversation turn...');
    const payload: IntegrationRequest = {
      userMessage: lastUserMsg.content,
      agentResponse: lastAiMsg.content,
      metadata: extractLLMMetadata(event),
    };

    await memoriClient.augmentation(payload);
    logger.info('Augmentation successful!');
  } catch (err) {
    logger.error(`Augmentation failed: ${err instanceof Error ? err.message : String(err)}`);
  } finally {
    logger.endSection('AUGMENTATION HOOK END');
  }
}
