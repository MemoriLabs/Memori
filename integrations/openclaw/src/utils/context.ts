import { OpenClawEvent, OpenClawContext } from '../types.js';

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
 * Throws an error if required context fields cannot be resolved.
 *
 * @param event - OpenClaw event object
 * @param ctx - OpenClaw context object
 * @param configuredEntityId - Hardcoded entity ID from plugin config
 * @returns Normalized context with entityId, sessionId, and provider
 * @throws Error If entityId, sessionId, or provider cannot be determined
 */
export function extractContext(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  configuredEntityId: string
): ExtractedContext {
  const entityId = configuredEntityId || event.userId;
  const sessionId = ctx.sessionKey || event.sessionId;
  const provider = ctx.messageProvider || event.messageProvider;

  if (!entityId)
    throw new Error(
      'Failed to extract context: Missing entityId (ensure plugin config or event.userId is set).'
    );

  if (!sessionId)
    throw new Error('Failed to extract context: Missing sessionId in OpenClaw context and event.');

  if (!provider)
    throw new Error(
      'Failed to extract context: Missing message provider in OpenClaw context and event.'
    );

  return {
    entityId,
    sessionId,
    provider,
  };
}
