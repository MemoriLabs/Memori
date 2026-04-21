# Memori TypeScript examples

Layout mirrors the Python examples under the repository root `examples/` directory:

| Python (`examples/…`) | TypeScript (`memori-ts/examples/…`) |
|------------------------|--------------------------------------|
| `sqlite/main.py` | `sqlite/main.ts` |
| `postgres/main.py` | `postgres/main.ts` |
| `cockroachdb/main.py` | `cockroachdb/main.ts` |
| `sqlite/rust_core_main.py` | `sqlite/rust_core_main.ts` |

MySQL uses the same three-turn quickstart as PostgreSQL (`mysql/main.ts`). MongoDB is not wired in the TS SDK yet.

**Cloud (no local DB)** lives in `cloud/main.ts`.

**ORM adapters** (TypeScript-only) are under `orm/`:

- `orm/drizzle-pg.ts` — Drizzle + PostgreSQL
- `orm/sequelize-mysql.ts` — Sequelize + MySQL
- `orm/typeorm-sqlite.ts` — TypeORM + SQLite
- `orm/mikro-sqlite.ts` — MikroORM + SQLite

## Run

From `memori-ts/`:

```bash
npm install
export OPENAI_API_KEY=...
npm run example:sqlite
```

Native BYODB examples require the Node bindings (`npm run sync-native`, which also runs before `build:dev`). That step needs **Rust** and **cargo**; see `memori-ts/README.md` under development.

See each folder’s `README.md` for required environment variables.
