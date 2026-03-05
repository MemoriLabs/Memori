[![Memori Labs](https://s3.us-east-1.amazonaws.com/images.memorilabs.ai/banner.png)](https://memorilabs.ai/)

<p align="center">
  <strong>The memory fabric for enterprise AI</strong>
</p>

<p align="center">
  <i>By default, OpenClaw agents forget everything between sessions. This plugin fixes that. It watches conversations, extracts what matters, and brings it back when relevant—automatically.</i>
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/@memorilabs/openclaw-memori">
    <img src="https://img.shields.io/npm/v/@memorilabs/openclaw-memori.svg" alt="NPM version">
  </a>
  <a href="https://www.npmjs.com/package/@memorilabs/openclaw-memori">
    <img src="https://img.shields.io/npm/dm/@memorilabs/openclaw-memori.svg" alt="NPM Downloads">
  </a>
  <a href="https://opensource.org/license/apache-2-0">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue" alt="License">
  </a>
  <a href="https://discord.gg/abD4eGym6v">
    <img src="https://img.shields.io/discord/1042405378304004156?logo=discord" alt="Discord">
  </a>
</p>

---

## Key Features

- **Auto-Recall:** Before the agent responds, the plugin searches Memori for memories that match the current context and injects them directly into the prompt.
- **Auto-Capture:** After the agent responds, the plugin securely sends the exchange to Memori to extract new facts, update stale ones, and merge duplicates.
- **Bulletproof Sanitization:** Automatically strips OpenClaw system metadata, internal timestamps, and thinking blocks to prevent context pollution and feedback loops.
- **Stateless & Thread-Safe:** A completely stateless architecture ensures zero memory leaks and 100% thread safety for multi-agent OpenClaw gateways.

## Getting Started

Run the following commands in your terminal to install and enable the plugin:

```bash
# 1. Install the plugin from npm
openclaw plugins install @memorilabs/openclaw-memori

# 2. Enable it in your workspace
openclaw plugins enable openclaw-memori

# 3. Restart the OpenClaw gateway
openclaw gateway restart
```

## Configuration

The plugin needs your Memori API key and an Entity ID to function. You can configure this via the OpenClaw CLI, your `openclaw.json` file, or environment variables.

### Option A: Via OpenClaw CLI (Recommended)

```bash
openclaw config set plugins.entries.openclaw-memori.config.apiKey "YOUR_MEMORI_API_KEY"
openclaw config set plugins.entries.openclaw-memori.config.entityId "your-app-user-id"
```

### Option B: Via `openclaw.json`

Add the following to your `~/.openclaw/openclaw.json` file:

```json
{
  "plugins": {
    "entries": {
      "openclaw-memori": {
        "enabled": true,
        "config": {
          "apiKey": "your-memori-api-key",
          "entityId": "your-app-user-id"
        }
      }
    }
  }
}
```

### Configuration Options

| Option     | Type     | Required | Description                                                                                         |
| ---------- | -------- | -------- | --------------------------------------------------------------------------------------------------- |
| `apiKey`   | `string` | **Yes**  | Your Memori API key.                                                                                |
| `entityId` | `string` | **Yes**  | The unique identifier for the entity (e.g., user, agent, or tenant) to attribute these memories to. |

## How It Works (The Hook Lifecycle)

This plugin integrates deeply with OpenClaw's event lifecycle to provide seamless memory without interfering with your agent's core logic:

1. **`before_prompt_build` (Recall):** When a user sends a message, the plugin intercepts the event, queries the Memori API, and safely prepends relevant memories to the agent's system context.
2. **`agent_end` (Capture):** Once the agent finishes generating its response, the plugin captures the final `user` and `assistant` messages, sanitizes them, and sends them to the Memori integration endpoint for long-term storage and entity mapping.

## Contributing

We welcome contributions from the community! Please see our [Contributing Guidelines](https://github.com/MemoriLabs/Memori/blob/main/CONTRIBUTING.md) for details on code style, standards, and submitting pull requests.

To build from source:

```bash
# Clone the repository
git clone https://github.com/memorilabs/openclaw-memori.git
cd openclaw-memori

# Install dependencies and build
npm install
npm run build

# Run formatting, linting, and type checking
npm run check
```

---

## Support

- **Documentation**: [https://memorilabs.ai/docs](https://memorilabs.ai/docs)
- **Discord**: [https://discord.gg/abD4eGym6v](https://discord.gg/abD4eGym6v)
- **Issues**: [GitHub Issues](https://github.com/memorilabs/openclaw-memori/issues)

---

## License

Apache 2.0 - see [LICENSE](https://github.com/MemoriLabs/Memori/blob/main/LICENSE)
