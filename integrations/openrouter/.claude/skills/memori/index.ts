#!/usr/bin/env bun

declare const process: {
  env: Record<string, string | undefined>;
  exit(code?: number): never;
  argv: string[];
};

const API_KEY = process.env.MEMORI_API_KEY;
const ENTITY_ID = process.env.MEMORI_ENTITY_ID;
const DEFAULT_PROJECT_ID = process.env.MEMORI_PROJECT_ID;

const BASE_URL = "https://staging-api.memorilabs.ai/v1";
const COLLECTOR_URL = "https://staging-collector.memorilabs.ai/v1";
const X_API_KEY = "c18b1022-7fe2-42af-ab01-b1f9139184f0";

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
    console.error("MEMORI_ENTITY_ID is required");
    process.exit(1);
  }
  return ENTITY_ID;
}

function parseArgs(argv: string[]): {
  command: string;
  flags: Record<string, string>;
} {
  const command = argv[0] ?? "";
  const flags: Record<string, string> = {};
  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i];
    if (
      arg.startsWith("--") &&
      i + 1 < argv.length &&
      !argv[i + 1].startsWith("--")
    ) {
      flags[arg.slice(2)] = argv[++i];
    }
  }
  return { command, flags };
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

async function get(url: string): Promise<unknown> {
  const res = await fetch(url, { headers: headers() });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

async function post(url: string, body: unknown): Promise<unknown> {
  const res = await fetch(url, {
    method: "POST",
    headers: headers(),
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  if (res.status === 204) return {};
  return res.json();
}

async function recall(flags: Record<string, string>): Promise<void> {
  requireApiKey();
  const entityId = requireEntityId();
  const source = flags.source;
  const signal = flags.signal;
  const projectId = flags.projectId ?? DEFAULT_PROJECT_ID;

  if (flags.sessionId && !projectId) {
    console.error("sessionId cannot be provided without projectId");
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

  const qs = buildQS({
    query: flags.query,
    entity_id: entityId,
    project_id: projectId,
    session_id: flags.sessionId,
    date_start: flags.dateStart,
    date_end: flags.dateEnd,
    source,
    signal,
  });

  const result = await get(`${BASE_URL}/agent/recall${qs}`);
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

async function recallSummary(flags: Record<string, string>): Promise<void> {
  requireApiKey();
  const projectId = flags.projectId ?? DEFAULT_PROJECT_ID;

  if (flags.sessionId && !projectId) {
    console.error("sessionId cannot be provided without projectId");
    process.exit(1);
  }

  const qs = buildQS({
    project_id: projectId,
    session_id: flags.sessionId,
    date_start: flags.dateStart,
    date_end: flags.dateEnd,
  });

  const result = await get(`${BASE_URL}/agent/recall/summary${qs}`);
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

async function advancedAugmentation(
  flags: Record<string, string>
): Promise<void> {
  requireApiKey();
  const entityId = requireEntityId();
  const { sessionId, userMessage, assistantMessage, model, summary } = flags;
  const projectId = flags.projectId ?? DEFAULT_PROJECT_ID;
  const processId = flags.processId ?? process.env.MEMORI_PROCESS_ID;
  const trace = parseJsonFlag("trace", flags.trace);

  if (!sessionId || !userMessage || !assistantMessage) {
    console.error(
      "advanced-augmentation requires --sessionId, --userMessage, and --assistantMessage"
    );
    process.exit(1);
  }

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
    ...(projectId ? { project: { id: projectId } } : {}),
    session: { id: sessionId },
  };

  await post(`${BASE_URL}/agent/conversation/turn`, turnPayload);

  const augPayload = {
    attribution,
    conversation: { messages },
    meta: {
      sdk: { lang: "javascript", version: flags.sdkVersion ?? "openrouter-skill" },
      framework: { provider: flags.frameworkProvider ?? null },
      llm: {
        model: {
          provider: flags.provider ?? "openrouter",
          sdk: { version: flags.providerSdkVersion ?? null },
          version: model ?? null,
        },
      },
      platform: { provider: flags.platform ?? "openrouter" },
      storage: {
        cockroachdb: parseBooleanFlag(flags.cockroachdb),
        dialect: flags.storageDialect ?? null,
      },
    },
    ...(projectId ? { project: { id: projectId } } : {}),
    session: { id: sessionId, summary: summary ?? null },
    trace,
  };

  await post(`${COLLECTOR_URL}/agent/augmentation`, augPayload);

  console.log(JSON.stringify({ success: true, augmentation: true }));
  process.exit(0);
}

async function compaction(flags: Record<string, string>): Promise<void> {
  requireApiKey();
  const projectId = flags.projectId ?? DEFAULT_PROJECT_ID;

  if (!projectId) {
    console.error("compaction requires --projectId or MEMORI_PROJECT_ID env var");
    process.exit(1);
  }

  const qs = buildQS({
    project_id: projectId,
    session_id: flags.sessionId,
    num_messages: flags.numMessages,
  });

  const result = await get(`${BASE_URL}/agent/compaction${qs}`);
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

async function feedback(flags: Record<string, string>): Promise<void> {
  requireApiKey();
  if (!flags.content) {
    console.error("feedback requires --content");
    process.exit(1);
  }

  await post(`${BASE_URL}/agent/feedback`, { content: flags.content });
  console.log(JSON.stringify({ success: true }));
  process.exit(0);
}

async function quota(): Promise<void> {
  requireApiKey();
  const result = await get(`${BASE_URL}/sdk/quota`);
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

async function signup(flags: Record<string, string>): Promise<void> {
  const email = flags.email;
  if (!email) {
    console.error("signup requires --email");
    process.exit(1);
  }

  const emailRegex = /^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$/;
  if (!emailRegex.test(email)) {
    console.error(`The email you provided "${email}" is not valid.`);
    process.exit(1);
  }

  const result = await post(`${BASE_URL}/sdk/account`, { email });
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

const { command, flags } = parseArgs(process.argv.slice(2));

console.error(`[memori] command="${command}" flags=${JSON.stringify(flags)}`);

try {
  if (command === "recall") {
    await recall(flags);
  } else if (command === "recall.summary") {
    await recallSummary(flags);
  } else if (command === "advanced-augmentation") {
    await advancedAugmentation(flags);
  } else if (command === "compaction") {
    await compaction(flags);
  } else if (command === "feedback") {
    await feedback(flags);
  } else if (command === "quota") {
    await quota();
  } else if (command === "signup") {
    await signup(flags);
  } else {
    console.error(
      `Unknown command: "${command}". Valid commands: recall, recall.summary, advanced-augmentation, compaction, feedback, quota, signup`
    );
    process.exit(1);
  }
} catch (e) {
  console.error((e as Error).message ?? String(e));
  process.exit(1);
}

export {};
