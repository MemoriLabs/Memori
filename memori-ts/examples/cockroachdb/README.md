# Memori + CockroachDB Example

Example showing how to use Memori with CockroachDB (same narrative as `examples/cockroachdb/main.py` in the Python SDK). The Node `pg` driver speaks the PostgreSQL wire protocol.

## Quick start

1. **Install dependencies** (from `memori-ts/`):

   ```bash
   npm install
   ```

2. **Set environment variables**:

   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export COCKROACHDB_CONNECTION_STRING=postgresql://user:password@host:26257/defaultdb?sslmode=require
   ```

3. **Run**:

   ```bash
   npm run example:cockroachdb
   ```
