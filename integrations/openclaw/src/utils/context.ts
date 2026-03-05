import { OpenClawEvent, OpenClawContext } from '../types.js';
import { DEFAULTS } from '../constants.js';

/**
 * Extracted context information from OpenClaw events
 */
export interface ExtractedContext {
  entityId: string;
  sessionId: string;
  provider: string;
}

/**
 * Extracts and normalizes context information from OpenClaw event and context objects.
 * Provides sensible defaults for missing values.
 *
 * @param event - OpenClaw event object
 * @param ctx - OpenClaw context object
 * @param configuredEntityId - Optional hardcoded entity ID from plugin config
 * @returns Normalized context with entityId, sessionId, and provider
 */
export function extractContext(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  configuredEntityId?: string
): ExtractedContext {
  return {
    entityId: configuredEntityId || event.userId || DEFAULTS.USER_ID,
    sessionId: ctx.sessionKey || event.sessionId || DEFAULTS.SESSION_ID,
    provider: ctx.messageProvider || event.messageProvider || DEFAULTS.PROVIDER,
  };
}
