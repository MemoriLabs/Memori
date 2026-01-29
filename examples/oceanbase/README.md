# Memori + OceanBase Example

Example showing how to use Memori with OceanBase (or SeekDB).

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export DATABASE_CONNECTION_STRING=mysql+oceanbase://root:@localhost:2881/memori_test?charset=utf8mb4
   ```

3. **Run the example**:
   ```bash
   uv run python main.py
   ```

## What This Example Demonstrates

- **OceanBase integration**: Connect to OceanBase or SeekDB using the pyobvector dialect
- **Automatic persistence**: Conversation messages are stored in OceanBase tables
- **Context preservation**: Memori injects relevant history into each LLM call
