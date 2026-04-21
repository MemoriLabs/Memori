# ORM examples (TypeScript)

These scripts mirror the **same three-turn quickstart** as `examples/sqlite/main.ts` and `examples/postgres/main.ts`, but pass different connection objects supported by the TypeScript storage adapters:

| File | Stack |
|------|--------|
| `drizzle-pg.ts` | Drizzle + `pg` pool |
| `sequelize-mysql.ts` | Sequelize + MySQL (connection URI) |
| `typeorm-sqlite.ts` | TypeORM + better-sqlite3 |
| `mikro-sqlite.ts` | MikroORM + SQLite |

Run from `memori-ts/`:

```bash
npm run drizzle
npm run sequelize
npm run typeorm
npm run mikro
```

PostgreSQL/MySQL examples expect `DATABASE_CONNECTION_STRING` where applicable; see `examples/postgres/README.md` and `examples/mysql/README.md`.
