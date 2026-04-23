import { OpenClawEvent, OpenClawContext } from '../types.js';

/**
 * Extracted context information from OpenClaw events
 */
export interface ExtractedContext {
  entityId: string;
  sessionId: string;
  provider: string;
  projectId?: string;
}

/**
 * Extracts and normalizes context information from OpenClaw event and context objects.
 * Throws an error if required context fields cannot be resolved.
 *
 * @param event - OpenClaw event object
 * @param ctx - OpenClaw context object
 * @param configuredEntityId - Hardcoded entity ID from plugin config
 * @param configuredProjectId - Optional project ID from plugin config
 * @returns Normalized context with entityId, sessionId, provider, and optional projectId
 * @throws Error If entityId, sessionId, or provider cannot be determined
 */
export function extractContext(
  event: OpenClawEvent,
  ctx: OpenClawContext,
  configuredEntityId: string,
  configuredProjectId?: string
): ExtractedContext {
  const sessionId = ctx.sessionKey || event.sessionId;
  const provider = ctx.messageProvider || event.messageProvider;

  if (!sessionId) {
    throw new Error('Failed to extract context: Missing sessionId in OpenClaw context and event.');
  }

  if (!provider) {
    throw new Error(
      'Failed to extract context: Missing message provider in OpenClaw context and event.'
    );
  }

  return {
    entityId: configuredEntityId,
    sessionId,
    provider,
    projectId: configuredProjectId,
  };
}
