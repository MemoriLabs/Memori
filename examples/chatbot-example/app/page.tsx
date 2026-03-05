'use client';

import { useChat } from '@ai-sdk/react';
import { DefaultChatTransport } from 'ai';
import { useCallback, useMemo, useState } from 'react';

function getMessageText(parts: { type: string; text?: string }[]): string {
  if (!parts?.length) return '';
  return parts
    .filter((p): p is { type: string; text: string } => p.type === 'text' && typeof p.text === 'string')
    .map((p) => p.text)
    .join('');
}

export default function Chat() {
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: '/api/chat',
        body: {
          entityId: 'Koushik',
          processId: 'chatbot-example',
          sessionId,
        },
      }),
    [sessionId]
  );

  const { messages, sendMessage, status, setMessages } = useChat({
    transport,
  });

  const handleNewConversation = useCallback(() => {
    setSessionId(crypto.randomUUID());
    setMessages([]);
  }, [setMessages]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || status === 'streaming') return;
    sendMessage({ text });
    setInput('');
  };

  return (
    <div className="flex min-h-screen flex-col bg-stone-50">
      <header className="border-b border-stone-200 bg-white px-4 py-3 shadow-sm">
        <div className="mx-auto flex max-w-2xl items-center justify-between">
          <h1 className="text-lg font-semibold text-stone-800">Memori Chat</h1>
          <button
            type="button"
            onClick={handleNewConversation}
            className="rounded-lg border border-stone-300 bg-white px-3 py-1.5 text-sm text-stone-600 hover:bg-stone-50"
          >
            New conversation
          </button>
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-2xl flex-1 flex-col gap-4 overflow-y-auto p-4">
        {messages.length === 0 && (
          <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center text-stone-500">
            <p className="text-sm">Say something — I’ll remember it for later.</p>
            <p className="text-xs">Try: &quot;My name is Alex and I like hiking.&quot;</p>
          </div>
        )}
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-[85%] rounded-2xl px-4 py-2.5 text-sm ${
                message.role === 'user'
                  ? 'bg-emerald-600 text-white'
                  : 'bg-white text-stone-800 shadow-md ring-1 ring-stone-200'
              }`}
            >
              <div className="whitespace-pre-wrap break-words">
                {getMessageText(message.parts ?? [])}
              </div>
            </div>
          </div>
        ))}
        {status === 'streaming' && (
          <div className="flex justify-start">
            <div className="max-w-[85%] animate-pulse rounded-2xl bg-white px-4 py-2.5 text-sm text-stone-400 shadow-md ring-1 ring-stone-200">
              ...
            </div>
          </div>
        )}
      </main>

      <form onSubmit={handleSubmit} className="border-t border-stone-200 bg-white p-4">
        <div className="mx-auto flex max-w-2xl gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type a message..."
            disabled={status === 'streaming'}
            className="flex-1 rounded-xl border border-stone-300 bg-stone-50 px-4 py-3 text-stone-800 placeholder-stone-400 focus:border-emerald-500 focus:outline-none focus:ring-2 focus:ring-emerald-500/20 disabled:opacity-60"
          />
          <button
            type="submit"
            disabled={!input.trim() || status === 'streaming'}
            className="rounded-xl bg-emerald-600 px-4 py-3 font-medium text-white hover:bg-emerald-700 disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </form>
    </div>
  );
}
