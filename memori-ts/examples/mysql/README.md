# Memori + MySQL Example

Example showing how to use Memori with MySQL. The conversation script matches the PostgreSQL and SQLite quickstarts in this repo.

## Quick start

1. **Install dependencies** (from `memori-ts/`):

   ```bash
   npm install
   ```

2. **Set environment variables**:

   ```bash
   export OPENAI_API_KEY=your_api_key_here
   export DATABASE_CONNECTION_STRING=mysql://memori:memori@localhost:3307/memori_test
   ```

3. **Run**:

   ```bash
   npm run example:mysql
   ```

## What this example demonstrates

- **MySQL integration** via `mysql2` and a connection URI.
- **Same three-turn “Paris / favorite color” narrative** as the other SQL quickstarts for easy comparison across SDKs.
