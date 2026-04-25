# Memori + TiDB Example

Example showing how to use Memori with TiDB / TiDB Cloud.

## Quick Start

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set environment variables**:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export DATABASE_CONNECTION_STRING=mysql+pymysql://user:password@host:4000/memori_db?charset=utf8mb4
   ```

   For **TiDB Cloud Serverless**, also enable TLS for the SQLAlchemy / PyMySQL
   connection:
   ```bash
   export DATABASE_USE_TLS=1
   ```

3. **Run the example**:
   ```bash
   uv run python main.py
   ```

## What This Example Demonstrates

- **TiDB integration**: Connect to TiDB or TiDB Cloud using a standard MySQL-compatible connection string
- **Automatic TiDB detection**: Memori auto-detects TiDB from `SELECT VERSION()` and routes it through the dedicated TiDB integration path
- **Automatic persistence**: Memori persists memory and conversation context in TiDB
- **Context preservation**: Memori injects relevant history into each LLM call
- **Serverless-ready TLS**: `DATABASE_USE_TLS=1` adds a CA-backed TLS config for TiDB Cloud Serverless
