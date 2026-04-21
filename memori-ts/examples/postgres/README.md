# Memori + PostgreSQL Example

Example showing how to use Memori with PostgreSQL (same flow as `examples/postgres/main.py` in the Python SDK).

## Quick start

1. **Install dependencies** (from `memori-ts/`):

   ```bash
   npm install
   ```

2. **Set environment variables**:

   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export DATABASE_CONNECTION_STRING=postgresql://user:password@localhost:5432/dbname
   ```

   Use a `postgresql://` URL suitable for the `pg` driver (not the `postgresql+psycopg://` style used by SQLAlchemy in Python).

3. **Run**:

   ```bash
   npm run example:postgres
   ```

## What this example demonstrates

- **PostgreSQL integration**: Connect to any PostgreSQL-compatible database the `pg` package supports.
- **Automatic persistence**: Memories are stored in your database via the BYODB path.
- **Context preservation**: Memori recalls relevant facts across the scripted conversation.
