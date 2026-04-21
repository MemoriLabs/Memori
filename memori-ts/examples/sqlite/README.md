# Memori + SQLite Example

Example showing how to use Memori with SQLite (same flow as `examples/sqlite/main.py` in the Python SDK).

## Quick start

1. **Install dependencies** (from `memori-ts/`):

   ```bash
   npm install
   ```

2. **Set environment variables**:

   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

3. **Run**:

   ```bash
   npm run example:sqlite
   ```

## What this example demonstrates

- **Automatic persistence**: Conversation turns are processed for long-term memory via the local Rust engine and your SQLite file (`memori.db`).
- **Context preservation**: Memori injects relevant memories into each LLM call when integrated via `llm.register`.
- **Portable**: The database file can be copied, backed up, or shared easily.

## Rust core smoke test

`rust_core_main.ts` mirrors `examples/sqlite/rust_core_main.py`: a shorter script that exercises BYODB + the native engine.
