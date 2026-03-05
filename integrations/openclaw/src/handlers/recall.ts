import { cleanText, isSystemMessage } from '../sanitizer.js';
import { OpenClawEvent, OpenClawContext, MemoriPluginConfig } from '../types.js';
import { RECALL_CONFIG } from '../constants.js';
import { extractContext, MemoriLogger, initializeMemoriClient } from '../utils/index.js';

export async function handleRecall(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  config: MemoriPluginConfig,
  logger: MemoriLogger
): Promise<{ prependContext: string } | undefined> {
  logger.section('RECALL HOOK START');

  const context = extractContext(event, ctx, config.entityId);
  logger.info(
    `EntityID: ${context.entityId} | SessionID: ${context.sessionId} | Provider: ${context.provider}`
  );

  const promptText = cleanText(event.prompt);

  if (
    !promptText ||
    promptText.length < RECALL_CONFIG.MIN_PROMPT_LENGTH ||
    isSystemMessage(promptText)
  ) {
    logger.info('Prompt too short or is a system message. Aborting recall.');
    return undefined;
  }

  const memoriClient = initializeMemoriClient(config.apiKey, context);

  try {
    logger.info('Executing SDK Recall...');

    const recallText = await memoriClient.recall(promptText);
    const hookReturn = recallText ? { prependContext: recallText } : undefined;

    if (hookReturn) {
      logger.info(`Successfully injected memory context.`);
    } else {
      logger.info('No relevant memories found.');
    }

    logger.endSection('RECALL HOOK END');
    return hookReturn;
  } catch (err) {
    logger.error(`Recall failed: ${err instanceof Error ? err.message : String(err)}`);
    return undefined;
  }
}
