import { OpenClawIntegration } from '@memorilabs/memori/integrations';
import { SessionData } from '../types.js';
import { ExtractedContext } from './context.js';

/**
 * Initializes and configures a Memori OpenClaw integration instance
 *
 * @param session - Session data containing Memori instance
 * @param context - Extracted context information
 * @returns Configured OpenClawIntegration instance
 */
export function initializeMemoriClient(
  session: SessionData,
  context: ExtractedContext
): OpenClawIntegration {
  const openclaw = session.memori.integrate(OpenClawIntegration);
  openclaw.setAttribution(context.entityId, context.provider);
  openclaw.setSession(context.sessionId);
  return openclaw;
}
