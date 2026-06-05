#!/usr/bin/env bun

/**
 * Memori Claude Code skill CLI.
 *
 * Usage (flags accept either --flag value or --flag=value):
 *
 *   bun .claude/skills/memori/index.ts recall [--projectId ID] [--sessionId ID] [--dateStart ISO] [--dateEnd ISO] [--source SOURCE --signal SIGNAL]
 *   bun .claude/skills/memori/index.ts recall.summary [--projectId ID] [--sessionId ID] [--dateStart ISO] [--dateEnd ISO]
 *   bun .claude/skills/memori/index.ts advanced-augmentation --userMessage "..." --assistantMessage "..." [--sessionId ID] [--projectId ID] [--model MODEL] [--summary "..."] [--trace '{"tools":[]}']
 *   bun .claude/skills/memori/index.ts compaction [--projectId ID] [--sessionId ID] [--numMessages 5]
 *   bun .claude/skills/memori/index.ts feedback --content "..."
 *   bun .claude/skills/memori/index.ts quota
 *   bun .claude/skills/memori/index.ts signup --email user@example.com
 *
 * Configuration (set in .claude/settings.local.json under "env", or as real
 * environment variables; a colocated .env file is also loaded as a fallback):
 *
 *   MEMORI_API_KEY    (required) Memori Cloud API key
 *   MEMORI_ENTITY_ID  (required) stable per-user/agent namespace string
 *   MEMORI_PROJECT_ID (optional) defaults to basename($CLAUDE_PROJECT_DIR)
 *   MEMORI_SESSION_ID (optional) used by writes (advanced-augmentation) and
 *                     by compaction; defaults to $CLAUDE_CODE_SESSION_ID.
 *                     Reads (recall, recall.summary) intentionally do NOT
 *                     default session_id so retrieval stays project-scoped
 *                     across new Claude Code sessions and /clear.
 *   MEMORI_PROCESS_ID (optional) process attribution
 *   MEMORI_API_URL_BASE (optional) Memori API base URL override
 *
 * --projectId and --sessionId override the defaults per call.
 *
 * On success: exits 0 and prints JSON to stdout.
 * On failure: exits 1 and prints error to stderr.
 */

declare const process: {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
  argv: string[];
};

declare const Bun: {
  file(input: string | URL): {
    exists(): Promise<boolean>;
    text(): Promise<string>;
  };
};

// Load colocated .env so the skill works regardless of cwd.
// Real env vars and Bun's cwd-.env loading both take precedence.
try {
  const envFile = Bun.file(new URL("./.env", import.meta.url));
  if (await envFile.exists()) {
    const text = await envFile.text();
    for (const rawLine of text.split("\n")) {
      const line = rawLine.trim();
      if (!line || line.startsWith("#")) continue;
      const eq = line.indexOf("=");
      if (eq === -1) continue;
      const key = line.slice(0, eq).trim();
      let value = line.slice(eq + 1).trim();
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1);
      }
      if (process.env[key] == null) process.env[key] = value;
    }
  }
} catch {
  // Non-fatal: fall through to env-var validation below.
}

function basename(path: string): string {
  const trimmed = path.replace(/[/\\]+$/, "");
  const lastSlash = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
  return lastSlash >= 0 ? trimmed.slice(lastSlash + 1) : trimmed;
}

const API_KEY = process.env.MEMORI_API_KEY;
const ENTITY_ID = process.env.MEMORI_ENTITY_ID;

// Dynamic defaults from Claude Code's auto-injected subprocess env.
// CLAUDE_PROJECT_DIR -> default Memori project ID (basename of workspace).
// CLAUDE_CODE_SESSION_ID -> default Memori session ID for writes and for
// compaction (which restores the current session). Read paths (recall,
// recall.summary) do NOT auto-apply this — see comments in those functions.
const CLAUDE_PROJECT_DIR = process.env.CLAUDE_PROJECT_DIR;
const CLAUDE_SESSION_ID = process.env.CLAUDE_CODE_SESSION_ID;

const DEFAULT_PROJECT_ID =
  process.env.MEMORI_PROJECT_ID ??
  (CLAUDE_PROJECT_DIR ? basename(CLAUDE_PROJECT_DIR) : undefined);

const DEFAULT_SESSION_ID = process.env.MEMORI_SESSION_ID ?? CLAUDE_SESSION_ID;

const PUBLIC_PROD_KEY = "96a7ea3e-11c2-428c-b9ae-5a168363dc80";
const PUBLIC_STAGING_KEY = "c18b1022-7fe2-42af-ab01-b1f9139184f0";

function env(name: string): string | undefined {
  const value = process.env[name]?.trim();
  return value || undefined;
}

function versionedBaseUrl(baseUrl: string): string {
  const trimmed = baseUrl.replace(/\/+$/, "");
  return trimmed.endsWith("/v1") ? trimmed : `${trimmed}/v1`;
}

const API_URL_BASE = env("MEMORI_API_URL_BASE");
const BASE_URL = versionedBaseUrl(API_URL_BASE ?? "https://api.memorilabs.ai");
const COLLECTOR_URL = versionedBaseUrl(
  API_URL_BASE ?? "https://collector.memorilabs.ai"
);
const X_API_KEY = API_URL_BASE ? PUBLIC_STAGING_KEY : PUBLIC_PROD_KEY;

type HttpMethod = "GET" | "POST";

type ToolTrace = {
  tools: Array<{
    name: string;
    args: Record<string, unknown>;
    result: unknown;
  }>;
};

const VALID_SOURCE_SIGNAL: Record<string, string> = {
  constraint: "discovery",
  decision: "commit",
  execution: "failure",
  fact: "verification",
  insight: "inference",
  instruction: "discovery",
  status: "update",
  strategy: "pattern",
  task: "result",
};

function parseJsonFlag(name: string, value: string | undefined): unknown {
  if (value == null || value === "") return null;
  try {
    return JSON.parse(value);
  } catch (e) {
    throw new Error(`Invalid JSON for --${name}: ${(e as Error).message}`);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function parseTraceFlag(value: string | undefined): ToolTrace {
  const parsed = parseJsonFlag("trace", value);
  if (parsed == null) return { tools: [] };
  if (!isRecord(parsed) || !Array.isArray(parsed.tools)) {
    throw new Error('--trace must be JSON shaped like {"tools":[]}');
  }

  for (const tool of parsed.tools) {
    if (!isRecord(tool)) {
      throw new Error("--trace tools entries must be objects");
    }
    if (typeof tool.name !== "string") {
      throw new Error("--trace tools entries require string name");
    }
    if (!isRecord(tool.args)) {
      throw new Error("--trace tools entries require object args");
    }
    if (!("result" in tool)) {
      throw new Error("--trace tools entries require result");
    }
  }

  return parsed as ToolTrace;
}

function parseBooleanFlag(value: string | undefined): boolean {
  return value === "1" || value === "true" || value === "yes";
}

function requireApiKey(): string {
  if (!API_KEY) {
    console.error("MEMORI_API_KEY is required");
    process.exit(1);
  }
  return API_KEY;
}

function requireEntityId(): string {
  if (!ENTITY_ID) {
    console.error(
      "MEMORI_ENTITY_ID is required. Pick any stable string that namespaces " +
        "memory for this user/agent (e.g. " +
        "hostname, or a generated UUID), then add it to " +
        '.claude/settings.local.json: { "env": { "MEMORI_ENTITY_ID": "<value>" } }. ' +
        "If no sensible default is available, ask the user."
    );
    process.exit(1);
  }
  return ENTITY_ID;
}

function requireProjectId(flags: Record<string, string>): string {
  const projectId = flags.projectId ?? DEFAULT_PROJECT_ID;
  if (!projectId) {
    console.error(
      "MEMORI_PROJECT_ID could not be resolved. It normally defaults to " +
        "basename($CLAUDE_PROJECT_DIR). Pass --projectId, or set MEMORI_PROJECT_ID " +
        "in .claude/settings.local.json under env."
    );
    process.exit(1);
  }
  return projectId;
}

function resolveSessionId(flags: Record<string, string>): string | undefined {
  return flags.sessionId ?? DEFAULT_SESSION_ID;
}

function requireSessionId(
  flags: Record<string, string>,
  command: string
): string {
  const sessionId = resolveSessionId(flags);
  if (!sessionId) {
    console.error(
      `${command} requires a session ID. It normally defaults to ` +
        "$CLAUDE_CODE_SESSION_ID. Pass --sessionId, or set MEMORI_SESSION_ID " +
        "in .claude/settings.local.json under env."
    );
    process.exit(1);
  }
  return sessionId;
}

function parseArgs(argv: string[]): {
  command: string;
  flags: Record<string, string>;
} {
  const command = argv[0] ?? "";
  const flags: Record<string, string> = {};
  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i];
    if (!arg.startsWith("--")) continue;

    const raw = arg.slice(2);
    const equalsIndex = raw.indexOf("=");
    if (equalsIndex >= 0) {
      flags[raw.slice(0, equalsIndex)] = raw.slice(equalsIndex + 1);
    } else if (i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      flags[raw] = argv[++i];
    } else {
      flags[raw] = "true";
    }
  }
  return { command, flags };
}

function requireFlags(
  flags: Record<string, string>,
  command: string,
  ...names: string[]
): void {
  const missing = names.filter((name) => !flags[name]);
  if (missing.length === 0) return;

  console.error(
    `${command} requires ${missing.map((name) => `--${name}`).join(", ")}`
  );
  process.exit(1);
}

function headers(): Record<string, string> {
  const result: Record<string, string> = {
    "Content-Type": "application/json",
    "X-Memori-API-Key": X_API_KEY,
  };
  if (API_KEY) result.Authorization = `Bearer ${API_KEY}`;
  return result;
}

function buildQS(params: Record<string, string | undefined>): string {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value != null && value !== "") qs.set(key, value);
  }
  const str = qs.toString();
  return str ? `?${str}` : "";
}

async function request(
  url: string,
  method: HttpMethod,
  body?: unknown
): Promise<unknown> {
  const res = await fetch(url, {
    method,
    headers: headers(),
    ...(body !== undefined ? { body: JSON.stringify(body) } : {}),
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Memori API error (${res.status} ${res.statusText}): ${body}`);
  }
  if (res.status === 204) return {};
  return res.json();
}

function get(url: string): Promise<unknown> {
  return request(url, "GET");
}

function post(url: string, body: unknown): Promise<unknown> {
  return request(url, "POST", body);
}

async function recall(flags: Record<string, string>): Promise<unknown> {
  requireApiKey();
  const entityId = requireEntityId();
  const projectId = requireProjectId(flags);
  const source = flags.source;
  const signal = flags.signal;

  if (flags.query != null) {
    console.error("recall does not support --query");
    process.exit(1);
  }

  if ((source == null) !== (signal == null)) {
    console.error("source and signal must be provided together");
    process.exit(1);
  }

  if (source != null && VALID_SOURCE_SIGNAL[source] !== signal) {
    console.error(
      `Invalid (source, signal) pair: (${source}, ${signal}). Expected signal "${VALID_SOURCE_SIGNAL[source]}" for source "${source}".`
    );
    process.exit(1);
  }

  // session_id narrows results to a single session on the agent recall API;
  // leave it unset by default so ambient recall stays project-scoped across
  // new Claude Code sessions and /clear. Pass --sessionId explicitly when
  // you want to scope to a specific session.
  const qs = buildQS({
    entity_id: entityId,
    project_id: projectId,
    session_id: flags.sessionId,
    date_start: flags.dateStart,
    date_end: flags.dateEnd,
    source,
    signal,
  });

  return get(`${BASE_URL}/agent/recall${qs}`);
}

async function recallSummary(flags: Record<string, string>): Promise<unknown> {
  requireApiKey();
  const projectId = requireProjectId(flags);

  // See note in recall: session_id is a narrowing filter, so the default
  // here is also project-scoped. Pass --sessionId for single-session summaries.
  const qs = buildQS({
    project_id: projectId,
    session_id: flags.sessionId,
    date_start: flags.dateStart,
    date_end: flags.dateEnd,
  });

  return get(`${BASE_URL}/agent/recall/summary${qs}`);
}

async function advancedAugmentation(
  flags: Record<string, string>
): Promise<unknown> {
  requireApiKey();
  const entityId = requireEntityId();
  const projectId = requireProjectId(flags);
  const { userMessage, assistantMessage, model, summary } = flags;
  const processId = flags.processId ?? process.env.MEMORI_PROCESS_ID;
  const trace = parseTraceFlag(flags.trace);

  requireFlags(
    flags,
    "advanced-augmentation",
    "userMessage",
    "assistantMessage"
  );

  const sessionId = requireSessionId(flags, "advanced-augmentation");

  const attribution = {
    entity: { id: entityId },
    ...(processId ? { process: { id: processId } } : {}),
  };

  const messages = [
    { role: "user", content: userMessage, type: "text", trace: null },
    {
      role: "assistant",
      content: assistantMessage,
      type: "text",
      trace,
    },
  ];

  const turnPayload = {
    attribution,
    messages,
    project: { id: projectId },
    session: { id: sessionId },
  };

  await post(`${BASE_URL}/agent/conversation/turn`, turnPayload);

  const augPayload = {
    attribution,
    conversation: { messages },
    meta: {
      sdk: { lang: "javascript", version: flags.sdkVersion ?? "claude-code-skill" },
      framework: { provider: flags.frameworkProvider ?? null },
      llm: {
        model: {
          provider: flags.provider ?? "claude-code",
          sdk: { version: flags.providerSdkVersion ?? null },
          version: model ?? null,
        },
      },
      platform: { provider: flags.platform ?? "claude-code" },
      storage: {
        cockroachdb: parseBooleanFlag(flags.cockroachdb),
        dialect: flags.storageDialect ?? null,
      },
    },
    project: { id: projectId },
    session: { id: sessionId, summary: summary ?? null },
    trace,
  };

  try {
    await post(`${COLLECTOR_URL}/agent/augmentation`, augPayload);
  } catch (e) {
    console.error(`[memori] collector augmentation failed (non-fatal): ${(e as Error).message}`);
  }

  return { success: true, augmentation: true };
}

async function compaction(flags: Record<string, string>): Promise<unknown> {
  requireApiKey();
  const projectId = requireProjectId(flags);

  const qs = buildQS({
    project_id: projectId,
    session_id: resolveSessionId(flags),
    num_messages: flags.numMessages,
  });

  return get(`${BASE_URL}/agent/compaction${qs}`);
}

async function feedback(flags: Record<string, string>): Promise<unknown> {
  requireApiKey();
  requireFlags(flags, "feedback", "content");

  await post(`${BASE_URL}/agent/feedback`, { content: flags.content });
  return { success: true };
}

async function quota(): Promise<unknown> {
  requireApiKey();
  return get(`${BASE_URL}/sdk/quota`);
}

async function signup(flags: Record<string, string>): Promise<unknown> {
  requireFlags(flags, "signup", "email");
  const email = flags.email;

  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  if (!emailRegex.test(email)) {
    console.error(`The email you provided "${email}" is not valid.`);
    process.exit(1);
  }

  return post(`${BASE_URL}/sdk/account`, { email });
}

const { command, flags } = parseArgs(process.argv.slice(2));

console.error(`[memori] command="${command}" flags=${JSON.stringify(flags)}`);

try {
  let result: unknown;

  if (command === "recall") {
    result = await recall(flags);
  } else if (command === "recall.summary") {
    result = await recallSummary(flags);
  } else if (command === "advanced-augmentation") {
    result = await advancedAugmentation(flags);
  } else if (command === "compaction") {
    result = await compaction(flags);
  } else if (command === "feedback") {
    result = await feedback(flags);
  } else if (command === "quota") {
    result = await quota();
  } else if (command === "signup") {
    result = await signup(flags);
  } else {
    console.error(
      `Unknown command: "${command}". Valid commands: recall, recall.summary, advanced-augmentation, compaction, feedback, quota, signup`
    );
    process.exit(1);
  }

  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
} catch (e) {
  console.error((e as Error).message ?? String(e));
  process.exit(1);
}

export {};
