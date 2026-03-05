# Memori OpenClaw Plugin

Official [Memori](https://memori.ai) long-term memory plugin for OpenClaw. Provides persistent, context-aware memory that survives across sessions and context compaction.

## Features

- 🧠 **Auto-Recall**: Automatically injects relevant memories before each agent response
- 💾 **Auto-Capture**: Automatically stores conversation facts after each interaction
- 🔄 **Session Management**: Efficient caching with automatic garbage collection
- 🎯 **Context-Aware**: Only surfaces memories relevant to the current query
- ⚡ **Performance**: Smart caching prevents duplicate API calls

## Installation

```bash
cd ~/.openclaw/extensions
git clone <your-repo-url> openclaw-memori
cd openclaw-memori
npm install
npm run build
```

## Configuration

Add to your `~/.openclaw/openclaw.json`:

```json
{
  "plugins": {
    "allow": ["openclaw-memori"],
    "slots": {
      "memory": "openclaw-memori"
    },
    "entries": {
      "openclaw-memori": {
        "enabled": true,
        "config": {
          "apiKey": "your-memori-api-key",
          "entityId": "optional-hardcoded-user-id"
        }
      }
    }
  }
}
```

### Environment Variables

Alternatively, set environment variables:

```bash
export MEMORI_API_KEY="your-memori-api-key"
export MEMORI_USER_ID="optional-user-id"  # Optional
```

### Configuration Options

| Option     | Type   | Required | Description                                                                       |
| ---------- | ------ | -------- | --------------------------------------------------------------------------------- |
| `apiKey`   | string | Yes      | Memori API key (falls back to `MEMORI_API_KEY` env var)                           |
| `entityId` | string | No       | Hardcoded entity ID for all memories (defaults to dynamic based on OpenClaw user) |

## How It Works

### Memory Recall (`before_prompt_build` hook)

1. User sends a message
2. Plugin queries Memori API for relevant memories
3. Memories are injected into conversation context
4. AI sees the memories and responds with full context

### Memory Capture (`agent_end` hook)

1. Agent completes response
2. Plugin sends the conversation turn to Memori
3. Memori extracts and stores important facts
4. Facts are deduplicated and merged with existing memories

## Architecture

```
User Message
    ↓
[Recall Hook] → Memori API → Relevant Memories
    ↓
[Inject into Context]
    ↓
AI Response
    ↓
[Capture Hook] → Memori API → Extract & Store Facts
```

## Development

```bash
# Install dependencies
npm install

# Build
npm run build

# Watch mode
npm run build:dev

# Lint
npm run lint

# Type check
npm run typecheck

# Run all checks
npm run check
```

## Performance

- **Session Caching**: 24-hour TTL with automatic garbage collection
- **API Call Deduplication**: Caches recall results to prevent duplicate API calls during double hook invocations
- **Minimal Latency**: ~2-5ms overhead beyond Memori API call time

## Troubleshooting

### Plugin not loading

Check OpenClaw logs for initialization messages:

```bash
tail -f ~/.openclaw/logs/gateway.log | grep Memori
```

### API key missing

Error: `[Memori] MEMORI_API_KEY is missing. Plugin disabled.`

Solution: Add `apiKey` to plugin config or set `MEMORI_API_KEY` environment variable.

### Memories not appearing

1. Check logs for "Successfully injected memory context"
2. Verify memories exist in your Memori account
3. Ensure queries are relevant to stored memories

## License

Apache-2.0

## Support

- **Memori Documentation**: https://docs.memori.ai
- **OpenClaw Documentation**: https://docs.openclaw.ai
- **Issues**: <your-repo-issues-url>
