import { createMemoriSignupTool } from './memori-signup.js';
import { createMemoriQuotaTool } from './memori-quota.js';
import { createMemoriRecallTool } from './memori-recall.js';
import { createMemoriRecallSummaryTool } from './memori-recall-summary.js';
import { createMemoriFeedbackTool } from './memori-feedback.js';
import { createMemoriCompactionTool } from './memori-compaction.js';
import type { ToolDeps } from './types.js';

export function registerUtilityTools(deps: ToolDeps): void {
  deps.api.registerTool(createMemoriSignupTool(deps));
}

export function registerAuthenticatedTools(deps: ToolDeps): void {
  deps.api.registerTool(createMemoriRecallTool(deps));
  deps.api.registerTool(createMemoriRecallSummaryTool(deps));
  deps.api.registerTool(createMemoriFeedbackTool(deps));
  deps.api.registerTool(createMemoriQuotaTool(deps));
  deps.api.registerTool(createMemoriCompactionTool(deps));
}

export type { ToolDeps };
