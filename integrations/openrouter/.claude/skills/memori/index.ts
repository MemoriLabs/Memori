#!/usr/bin/env bun

const API_KEY = process.env.MEMORI_API_KEY;
const ENTITY_ID = process.env.MEMORI_ENTITY_ID;
const DEFAULT_PROJECT_ID = process.env.MEMORI_PROJECT_ID;

if (!API_KEY || !ENTITY_ID) {
  console.error("MEMORI_API_KEY and MEMORI_ENTITY_ID are required");
  process.exit(1);
}

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

// --- arg parsing ---

function parseArgs(argv: string[]): { command: string; flags: Record<string, string> } {
  const command = argv[0] ?? "";
  const flags: Record<string, string> = {};
  for (let i = 1; i < argv.length; i++) {
    const arg = argv[i];
    if (arg.startsWith("--") && i + 1 < argv.length && !argv[i + 1].startsWith("--")) {
      flags[arg.slice(2)] = argv[++i];
    }
  }
  return { command, flags };
}

// --- http helpers ---

function headers(): Record<string, string> {
  return {
    "Content-Type": "application/json",
    "X-Memori-API-Key": X_API_KEY,
    Authorization: `Bearer ${API_KEY}`,
  };
}

function buildQS(params: Record<string, string | undefined>): string {
  const qs = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== "") qs.set(k, v);
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

// --- commands ---

async function recall(flags: Record<string, string>) {
  const source = flags.source;
  const signal = flags.signal;

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
    entity_id: ENTITY_ID,
    project_id: flags.projectId ?? DEFAULT_PROJECT_ID,
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

async function recallSummary(flags: Record<string, string>) {
  const qs = buildQS({
    project_id: flags.projectId ?? DEFAULT_PROJECT_ID,
    session_id: flags.sessionId,
    date_start: flags.dateStart,
    date_end: flags.dateEnd,
  });

  const result = await get(`${BASE_URL}/agent/recall/summary${qs}`);
  console.log(JSON.stringify(result, null, 2));
  process.exit(0);
}

async function capture(flags: Record<string, string>) {
  const { sessionId, userMessage, assistantMessage, model } = flags;
  const projectId = flags.projectId ?? DEFAULT_PROJECT_ID;

  if (!sessionId || !userMessage || !assistantMessage) {
    console.error("capture requires --sessionId, --userMessage, and --assistantMessage");
    process.exit(1);
  }

  const attribution = {
    entity: { id: ENTITY_ID },
  };

  const messages = [
    { role: "user", content: userMessage, type: "text", trace: null },
    { role: "assistant", content: assistantMessage, type: "text", trace: null },
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
      attribution,
      sdk: { lang: "javascript", version: "openrouter-skill" },
      framework: { provider: null },
      llm: {
        model: {
          provider: "openrouter",
          sdk: { version: null },
          version: model ?? null,
        },
      },
      platform: { provider: "openrouter" },
      storage: { cockroachdb: false, dialect: null },
    },
    ...(projectId ? { project: { id: projectId } } : {}),
    session: { id: sessionId, summary: null },
    trace: null,
  };

  post(`${COLLECTOR_URL}/agent/augmentation`, augPayload).catch(() => {});

  console.log(JSON.stringify({ success: true }));
  process.exit(0);
}

async function compaction(flags: Record<string, string>) {
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

async function feedback(flags: Record<string, string>) {
  if (!flags.content) {
    console.error("feedback requires --content");
    process.exit(1);
  }

  await post(`${BASE_URL}/agent/feedback`, { content: flags.content });
  console.log(JSON.stringify({ success: true }));
  process.exit(0);
}

// --- dispatch ---

const { command, flags } = parseArgs(process.argv.slice(2));

console.error(`[memori] command="${command}" flags=${JSON.stringify(flags)}`);

try {
  if (command === "recall") {
    await recall(flags);
  } else if (command === "recall.summary") {
    await recallSummary(flags);
  } else if (command === "capture") {
    await capture(flags);
  } else if (command === "compaction") {
    await compaction(flags);
  } else if (command === "feedback") {
    await feedback(flags);
  } else {
    console.error(`Unknown command: "${command}". Valid commands: recall, recall.summary, capture, compaction, feedback`);
    process.exit(1);
  }
} catch (e) {
  console.error((e as Error).message ?? String(e));
  process.exit(1);
}
export { };

