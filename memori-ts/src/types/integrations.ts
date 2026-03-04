import { Config } from '../core/config.js';
import { SessionManager } from '../core/session.js';
import { RecallEngine } from '../engines/recall.js';
import { PersistenceEngine } from '../engines/persistence.js';
import { AugmentationEngine } from '../engines/augmentation.js';
import type { OpenClawIntegration } from '../integrations/openclaw.js';

export interface MemoriCore {
  recall: RecallEngine;
  persistence: PersistenceEngine;
  augmentation: AugmentationEngine;
  config: Config;
  session: SessionManager;
}

export interface IntegrationRequest {
  userMessage: string;
  agentResponse: string;
  metadata?: IntegrationMetadata;
}

export interface IntegrationMetadata {
  provider: string | null | undefined;
  model: string | null | undefined;
  sdkVersion: string | null | undefined;
  integrationSdkVersion: string | null | undefined;
  platform: string | null | undefined;
}

export type SupportedIntegration = OpenClawIntegration;
export type IntegrationConstructor<T extends SupportedIntegration> = new (core: MemoriCore) => T;
