import { Type } from "@sinclair/typebox";
import type { AddOptions } from "../types.ts";
import { isSubagentSession } from "../isolation.ts";
import { isNoiseMessage, stripNoiseFromContent } from "../filtering.ts";
// v3.0.0: resolveCategories/ttlToExpirationDate removed - expiration_date/immutable no longer supported
import type { ToolDeps } from "./index.ts";

export function createMemoryAddTool(deps: ToolDeps) {
  const { api, provider, resolveUserId, getCurrentSessionId, buildAddOptions, buildSearchOptions, skillsActive } = deps;

  return {
    name: "memori_recall",
    label: "Recall Memory",
    description: "Explicitly fetch relevant memories from Memori using filters like date, project, session, signal, and source.",
    parameters: Type.Object({
      dateStart: Type.Optional(Type.String({ description: "ISO 8601 date string to filter memories created on or after this time" })),
      dateEnd: Type.Optional(Type.String({ description: "ISO 8601 date string to filter memories created on or before this time" })),
      projectId: Type.Optional(Type.String({ description: "Filter to a specific project. Defaults to the current project." })),
      sessionId: Type.Optional(Type.String({ description: "Filter to a specific session. Requires projectId to also be provided." })),
      signal: Type.Optional(Type.String({ description: "Filter to a specific fact signal (e.g., system, user, derived)" })),
      source: Type.Optional(Type.String({ description: "Filter to a specific source origin" })),
    }),

    async execute(_toolCallId: string, params: {
      dateStart?: string;
      dateEnd?: string;
      projectId?: string;
      sessionId?: string;
      signal?: string;
      source?: string;
    }) {
      const p = params as {
        dateStart?: string;
        dateEnd?: string;
        projectId?: string;
        sessionId?: string;
        signal?: string;
        source?: string;
      };

      const start = Date.now();
      try {
        const currentSessionId = getCurrentSessionId();

        if (isSubagentSession(currentSessionId)) {
          return { content: [{ type: "text", text: "Memory storage is not available in subagent sessions." }], details: { error: "subagent_blocked" } };
        }

        const uid = resolveUserId({ agentId: p.agentId, userId: p.userId });
        const runId = !(p.longTerm ?? true) && currentSessionId ? currentSessionId : undefined;

        if (skillsActive) {
          const rawMetadata = p.metadata;
          const category = p.category ?? rawMetadata?.category as string | undefined;
          const importance = p.importance ?? rawMetadata?.importance as number | undefined;
          const parsedMetadata: Record<string, unknown> = {
            ...(rawMetadata ?? {}),
            ...(category && { category }),
            ...(importance !== undefined && { importance }),
          };

          const addOpts: AddOptions = {
            user_id: uid, source: "OPENCLAW", infer: false,
            deduced_memories: allFacts, metadata: parsedMetadata ?? {},
          };
          if (runId) addOpts.run_id = runId;

          const result = await provider.add([{ role: "user", content: allFacts.join("\n") }], addOpts);
          const count = result.results?.length ?? 0;
          api.logger.info(`openclaw-mem0: stored ${count} memor${count === 1 ? "y" : "ies"} (infer=false, category=${category ?? "none"})`);

          deps.captureToolEvent("memory_add", { success: true, latency_ms: Date.now() - start, fact_count: allFacts.length, mode: "skills" });
          return {
            content: [{ type: "text", text: `Stored ${allFacts.length} fact(s) [${category ?? "uncategorized"}]: ${allFacts.map(f => `"${f.slice(0, 60)}${f.length > 60 ? "..." : ""}"`).join(", ")}` }],
            details: { action: "stored", mode: "skills", category, factCount: allFacts.length, results: result.results },
          };
        }

        const combinedText = allFacts.join("\n");

        const result = await provider.add([{ role: "user", content: combinedText }], buildAddOptions(uid, runId, currentSessionId));
        const added = result.results?.filter((r) => r.event === "ADD") ?? [];
        const updated = result.results?.filter((r) => r.event === "UPDATE") ?? [];
        const summary = [];
        if (added.length > 0) summary.push(`${added.length} added`);
        if (updated.length > 0) summary.push(`${updated.length} updated`);
        if (summary.length === 0) summary.push("No new memories extracted");

        deps.captureToolEvent("memory_add", { success: true, latency_ms: Date.now() - start, fact_count: allFacts.length });
        return {
          content: [{ type: "text", text: `Stored: ${summary.join(", ")}. ${result.results?.map((r) => `[${r.event}] ${r.memory}`).join("; ") ?? ""}` }],
          details: { action: "stored", results: result.results },
        };
      } catch (err) {
        deps.captureToolEvent("memory_add", { success: false, latency_ms: Date.now() - start, error: String(err) });
        return { content: [{ type: "text", text: `Memory add failed: ${String(err)}` }], details: { error: String(err) } };
      }
    },
  };
}