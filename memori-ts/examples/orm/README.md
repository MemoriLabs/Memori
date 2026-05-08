# ORM examples (TypeScript)

Memori works with any ORM — you don't need a special adapter. Your ORM already creates a raw connection pool; pass that same pool to Memori and both coexist on the same connection with no conflict.

```typescript
// Drizzle + PostgreSQL
import { drizzle } from 'drizzle-orm/node-postgres';
import pg from 'pg';
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });
const db = drizzle(pool);
const mem = new Memori({ conn: () => pool });

// Sequelize + MySQL
import mysql from 'mysql2';
const mysqlPool = mysql.createPool({ uri: process.env.DATABASE_URL });
const mem = new Memori({ conn: () => mysqlPool });

// MikroORM + PostgreSQL
import pg from 'pg';
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });
const mem = new Memori({ conn: () => pool });
```

Your ORM handles your application queries; Memori manages its own tables — same pool, no conflict.
