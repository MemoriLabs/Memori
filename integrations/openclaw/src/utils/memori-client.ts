import { Memori } from '@memorilabs/memori';
import { OpenClawIntegration } from '@memorilabs/memori/integrations';
import { ExtractedContext } from './context.js';

/**
 * Initializes and configures a Memori OpenClaw integration instance
 *
 * @param apiKey - Memori API key
 * @param context - Extracted context information
 * @returns Configured OpenClawIntegration instance
 */
export function initializeMemoriClient(
  apiKey: string,
  context: ExtractedContext
): OpenClawIntegration {
  const memori = new Memori();
  memori.config.apiKey = apiKey;

  const openclaw = memori.integrate(OpenClawIntegration);
  openclaw
    .scope(context.sessionId, context.projectId)
    .attribution(context.entityId, context.provider);

  return openclaw;
}

/**
 * Creates a minimal Memori client scoped only to an entity, with no session or project
 * context pre-set. Intended for use in tool execute handlers where OpenClaw does not
 * reliably provide session context — callers supply projectId/sessionId as explicit
 * parameters instead.
 *
 * @param apiKey - Memori API key
 * @param entityId - Entity ID for attribution
 * @returns Configured OpenClawIntegration instance
 */
export function createRecallClient(apiKey: string, entityId: string): OpenClawIntegration {
  const memori = new Memori();
  memori.config.apiKey = apiKey;

  const openclaw = memori.integrate(OpenClawIntegration);
  openclaw.attribution(entityId);

  return openclaw;
}
