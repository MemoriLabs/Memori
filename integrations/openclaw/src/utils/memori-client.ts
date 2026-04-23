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
  openclaw.scope(context.sessionId, context.projectId).attribution(context.entityId, context.provider);

  return openclaw;
}
