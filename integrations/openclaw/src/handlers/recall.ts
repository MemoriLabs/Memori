import type { OpenClawPluginApi } from 'openclaw/plugin-sdk';
import { getOrCreateSession } from '../manager.js';
import { cleanText, isSystemMessage } from '../sanitizer.js';
import { OpenClawEvent, OpenClawContext } from '../types.js';
import { RECALL_CONFIG } from '../constants.js';
import { extractContext } from '../utils/context.js';
import { MemoriLogger } from '../utils/logger.js';
import { initializeMemoriClient } from '../utils/memori-client.js';

/**
 * Handles memory recall before prompt building.
 * Fetches relevant memories from Memori and injects them into the conversation context.
 *
 * @param event - OpenClaw event containing the user's prompt
 * @param ctx - OpenClaw context with session information
 * @param api - Plugin API for logging and hooks
 * @param apiKey - Memori API key
 * @param configuredEntityId - Optional hardcoded entity ID
 * @returns Hook result with prependContext containing memories, or undefined
 */
export async function handleRecall(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  api: OpenClawPluginApi,
  apiKey: string,
  configuredEntityId: string | undefined
) {
  const logger = new MemoriLogger(api);
  logger.section('RECALL HOOK START');

  const context = extractContext(event, ctx, configuredEntityId);
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
    return;
  }

  const session = getOrCreateSession(context.entityId, apiKey);

  // Check cache
  if (session.lastPrompt === promptText) {
    logger.info('Cache hit! Returning previous recall block.');
    return session.lastRecallBlock;
  }

  const memoriClient = initializeMemoriClient(session, context);

  try {
    logger.info('Executing SDK Recall...');

    const recallText = await memoriClient.recall(promptText);
    const hookReturn = recallText ? { prependContext: recallText } : undefined;

    if (hookReturn) {
      logger.info(`Successfully injected memory context.`);
    } else {
      logger.info('No relevant memories found.');
    }

    // Update cache
    session.lastPrompt = promptText;
    session.lastRecallBlock = hookReturn;

    logger.endSection('RECALL HOOK END');
    return hookReturn;
  } catch (err) {
    logger.error(`Recall failed: ${err instanceof Error ? err.message : String(err)}`);

    // Clear cache on error to allow retry
    session.lastPrompt = undefined;
    session.lastRecallBlock = undefined;

    return undefined;
  }
}
