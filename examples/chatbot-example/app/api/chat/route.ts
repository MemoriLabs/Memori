import { streamText, convertToModelMessages } from 'ai';
import { openai } from '@ai-sdk/openai';
import { Memori } from '@memorilabs/memori';
import type { UIMessage } from 'ai';

const MODEL = 'gpt-4o-mini';

function getLastUserMessageText(messages: UIMessage[]): string | null {
  const lastUser = [...messages].reverse().find((m) => m.role === 'user');
  if (!lastUser || !lastUser.parts) return null;
  const textParts = lastUser.parts.filter(
    (p): p is { type: 'text'; text: string } => p.type === 'text'
  );
  return textParts.map((p) => p.text).join('\n').trim() || null;
}

function buildMemoriContext(facts: { content: string; score: number; dateCreated?: string }[], threshold: number): string {
  const relevant = facts.filter((f) => f.score >= threshold);
  if (relevant.length === 0) return '';
  const lines = relevant.map(
    (f) => `- ${f.content}${f.dateCreated ? ` . Stated at ${f.dateCreated}` : ''}`
  );
  return `\n\n<memori_context>\nOnly use the relevant context if it is relevant to the user's query. Relevant context about the user:\n${lines.join('\n')}\n</memori_context>`;
}

export async function POST(req: Request) {
  const body = await req.json();
  const messages: UIMessage[] = body.messages ?? [];
  const entityId = body.entityId ?? 'koushik';
  const processId = body.processId ?? 'chatbot-example';
  const sessionId = body.sessionId as string | undefined;

  const lastUserText = getLastUserMessageText(messages);
  if (!lastUserText) {
    return new Response(JSON.stringify({ error: 'No user message' }), {
      status: 400,
    });
  }

  const mem = new Memori();
  mem.attribution(entityId, processId);
  if (sessionId) mem.setSession(sessionId);

  let systemPrompt = `You are a warm, attentive assistant. You remember what the user tells you and refer to it naturally in later messages. Be concise and conversational.`;
  try {
    const facts = await mem.recall(lastUserText);
    const context = buildMemoriContext(facts, mem.config.recallRelevanceThreshold);
    if (context) systemPrompt += context;
  } catch (e) {
    console.warn('Memori recall failed:', e);
  }

  const modelMessages = await convertToModelMessages(messages);

  const result = streamText({
    model: openai(MODEL),
    messages: modelMessages,
    system: systemPrompt,
    onFinish: async ({ text }) => {
      try {
        await mem.recordTurn(lastUserText, text, { model: MODEL });
      } catch (e) {
        console.warn('Memori recordTurn failed:', e);
      }
    },
  });

  return result.toUIMessageStreamResponse();
}
