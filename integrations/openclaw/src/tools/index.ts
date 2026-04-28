import { createMemoriRecallTool } from './memori-recall.js';
import { createMemoriRecallSummaryTool } from './memori-recall-summary.js';
import type { ToolDeps } from './types.js';

export function registerAllTools(deps: ToolDeps): void {
  deps.api.registerTool(createMemoriRecallTool(deps));
  deps.api.registerTool(createMemoriRecallSummaryTool(deps));
}

export type { ToolDeps };
