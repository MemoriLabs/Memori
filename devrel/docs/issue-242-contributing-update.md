# Issue #242 Draft: Windows + Docker Compose Docs Update

This draft is written to be pasted into the Memori repo `CONTRIBUTING.md` in the Development Setup section. It adds a Windows-specific path and documents Docker Compose services, ports, and credentials based on the current `docker-compose.yml`.

---

## Insert after “Quick Start (Docker)”

### Windows Setup (PowerShell + Docker Desktop)

You can run the Docker workflow without `make` on Windows. Docker Desktop includes `docker compose`.

```powershell
# Copy the example environment file
Copy-Item .env.example .env

# Start the environment (builds containers)
docker compose up -d --build

# Enter the development container
# (If this hangs, run it from WSL or use Git Bash with winpty)
docker compose exec dev /bin/bash
```

If you prefer, use WSL2 (recommended for best Linux compatibility):

```bash
# In WSL2
cp .env.example .env

docker compose up -d --build

docker compose exec dev /bin/bash
```

**Make-free equivalents**

If you don’t have GNU Make installed, use these direct commands:

```bash
docker compose up -d --build     # make dev-up

docker compose exec dev /bin/bash # make dev-shell

docker compose down              # make dev-down
```

---

## Add a “Docker Compose Services” section

The services below are started by `docker compose up -d --build` (or `make dev-up`).

| Service | Host Port | Container Port | Default Credentials | Notes |
| --- | --- | --- | --- | --- |
| `dev` | — | — | Uses `.env` | Dev container; run `docker compose exec dev /bin/bash` |
| `postgres` | 5432 | 5432 | user `memori` / pass `memori` / db `memori_test` | `postgresql://memori:memori@localhost:5432/memori_test` |
| `mysql` | 3307 | 3306 | user `memori` / pass `memori` / db `memori_test` (root pass `memori`) | `mysql+pymysql://memori:memori@localhost:3307/memori_test` |
| `mongodb` | 27017 | 27017 | user `memori` / pass `memori` / db `memori_test` | `mongodb://memori:memori@localhost:27017/memori_test?authSource=admin` |
| `mongo-express` | 8081 | 8081 | basic auth `admin` / `pass` | http://localhost:8081 |
| `oceanbase` | 2882 | 2881 | user `root` / pass *(empty)* | `mysql+oceanbase://root:@localhost:2882/memori_test?charset=utf8mb4` |
| `oracle` | 1521 | 1521 | user `system` / pass `memori` | `oracle+oracledb://system:memori@localhost:1521/?service_name=FREEPDB1` |

**Inside the `dev` container**, use container hostnames from `docker-compose.yml`:
- `postgresql://memori:memori@postgres:5432/memori_test`
- `mysql+pymysql://memori:memori@mysql:3306/memori_test`
- `mongodb://memori:memori@mongodb:27017/memori_test?authSource=admin`
- `mysql+oceanbase://root:@oceanbase:2881/memori_test?charset=utf8mb4`
- `oracle+oracledb://system:memori@oracle:1521/?service_name=FREEPDB1`

---

## Optional: clarify the `.env` expectations

Add under “Quick Start (Docker)”:

```
# Required for integration tests
# OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, XAI_API_KEY, NEBIUS_API_KEY
```

This mirrors the current `.env.example` and makes the required keys explicit.
