// API Route: /api/glean/chat
// This proxies requests to Glean's API and handles streaming

import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'edge';

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();

    if (!messages || !Array.isArray(messages)) {
      return NextResponse.json(
        { error: 'Messages array required' },
        { status: 400 }
      );
    }

    const gleanApiKey = process.env.GLEAN_API_KEY;
    const gleanInstance = process.env.GLEAN_INSTANCE;

    if (!gleanApiKey || !gleanInstance) {
      return NextResponse.json(
        { error: 'Glean credentials not configured' },
        { status: 500 }
      );
    }

    // Call Glean Chat API with streaming
    const response = await fetch(
      `https://${gleanInstance}-be.glean.com/api/v1/chat`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${gleanApiKey}`,
          'Content-Type': 'application/json',
          'X-Glean-Source': 'custom-chat-demo'
        },
        body: JSON.stringify({
          messages: messages.map((m: any) => ({
            role: m.role,
            content: m.content
          })),
          stream: true,
          options: {
            includeCitations: true,
            searchContext: true
          }
        })
      }
    );

    if (!response.ok) {
      const error = await response.text();
      console.error('Glean API error:', error);
      return NextResponse.json(
        { error: 'Failed to get response from Glean' },
        { status: response.status }
      );
    }

    // Stream the response back to the client
    const encoder = new TextEncoder();
    const stream = new ReadableStream({
      async start(controller) {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder();

        if (!reader) {
          controller.close();
          return;
        }

        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            const chunk = decoder.decode(value);
            const lines = chunk.split('\n').filter(line => line.trim());

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                const data = line.slice(6);
                
                // Forward the SSE data to the client
                controller.enqueue(
                  encoder.encode(`data: ${data}\n\n`)
                );
              }
            }
          }
        } catch (error) {
          console.error('Streaming error:', error);
        } finally {
          controller.close();
        }
      }
    });

    return new NextResponse(stream, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    });

  } catch (error) {
    console.error('API route error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}
