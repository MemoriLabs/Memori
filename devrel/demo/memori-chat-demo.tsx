// Glean Chat Demo - Next.js Component
// This demonstrates integrating Glean's Chat API with streaming responses

'use client';

import { useState } from 'react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  citations?: Citation[];
}

interface Citation {
  title: string;
  url: string;
  snippet: string;
}

export default function GleanChatDemo() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { role: 'user', content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Call your API route that proxies to Glean
      const response = await fetch('/api/glean/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, userMessage].map(m => ({
            role: m.role,
            content: m.content
          }))
        })
      });

      if (!response.ok) throw new Error('Failed to get response');

      // Handle streaming response
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      let assistantMessage: Message = { role: 'assistant', content: '', citations: [] };
      
      setMessages(prev => [...prev, assistantMessage]);

      while (reader) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n').filter(line => line.trim());

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.content) {
              assistantMessage.content += data.content;
              setMessages(prev => {
                const updated = [...prev];
                updated[updated.length - 1] = { ...assistantMessage };
                return updated;
              });
            }

            if (data.citations) {
              assistantMessage.citations = data.citations;
            }
          }
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.'
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto p-4">
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6 rounded-t-lg text-white">
        <h1 className="text-2xl font-bold">Glean AI Assistant</h1>
        <p className="text-sm opacity-90">Ask anything about your company knowledge</p>
      </div>

      <div className="flex-1 overflow-y-auto bg-gray-50 p-6 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 mt-20">
            <div className="text-6xl mb-4">🔍</div>
            <h2 className="text-xl font-semibold mb-2">Start a conversation</h2>
            <p>Ask about company policies, projects, people, or anything else!</p>
            <div className="mt-6 space-y-2">
              <button
                onClick={() => setInput("What are our Q4 objectives?")}
                className="block w-full max-w-md mx-auto p-3 bg-white rounded-lg shadow hover:shadow-md transition text-left"
              >
                📊 What are our Q4 objectives?
              </button>
              <button
                onClick={() => setInput("Who works on the AI team?")}
                className="block w-full max-w-md mx-auto p-3 bg-white rounded-lg shadow hover:shadow-md transition text-left"
              >
                👥 Who works on the AI team?
              </button>
              <button
                onClick={() => setInput("How do I submit an expense report?")}
                className="block w-full max-w-md mx-auto p-3 bg-white rounded-lg shadow hover:shadow-md transition text-left"
              >
                💰 How do I submit an expense report?
              </button>
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <div
            key={index}
            className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-3xl rounded-lg p-4 ${
                message.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : 'bg-white shadow-md'
              }`}
            >
              <div className="whitespace-pre-wrap">{message.content}</div>
              
              {message.citations && message.citations.length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="text-sm font-semibold mb-2 text-gray-600">Sources:</div>
                  {message.citations.map((citation, idx) => (
                    <a
                      key={idx}
                      href={citation.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="block mb-2 p-2 bg-gray-50 rounded hover:bg-gray-100 transition"
                    >
                      <div className="font-medium text-blue-600">{citation.title}</div>
                      <div className="text-xs text-gray-500 truncate">{citation.url}</div>
                    </a>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white rounded-lg p-4 shadow-md">
              <div className="flex space-x-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="bg-white border-t p-4">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && !isLoading && sendMessage()}
            placeholder="Ask anything..."
            className="flex-1 px-4 py-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            disabled={isLoading}
          />
          <button
            onClick={sendMessage}
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition font-semibold"
          >
            Send
          </button>
        </div>
        <div className="text-xs text-gray-500 mt-2">
          Powered by Glean API • Answers are based on your company's knowledge base
        </div>
      </div>
    </div>
  );
}
