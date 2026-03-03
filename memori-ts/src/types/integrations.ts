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

export type SupportedIntegration = OpenClawIntegration;
export type IntegrationConstructor<T extends SupportedIntegration> = new (core: MemoriCore) => T;
