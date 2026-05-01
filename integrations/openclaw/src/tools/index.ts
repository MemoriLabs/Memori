import { createMemoriSignupTool } from './memori-signup.js';
import { createMemoriQuotaTool } from './memori-quota.js';
import { createMemoriRecallTool } from './memori-recall.js';
import { createMemoriRecallSummaryTool } from './memori-recall-summary.js';
import { createMemoriFeedbackTool } from './memori-feedback.js';
import type { ToolDeps } from './types.js';

export function registerAllTools(deps: ToolDeps): void {
  deps.api.registerTool(createMemoriSignupTool(deps));
  deps.api.registerTool(createMemoriQuotaTool(deps));
  deps.api.registerTool(createMemoriRecallTool(deps));
  deps.api.registerTool(createMemoriRecallSummaryTool(deps));
  deps.api.registerTool(createMemoriFeedbackTool(deps));
}

export type { ToolDeps };
