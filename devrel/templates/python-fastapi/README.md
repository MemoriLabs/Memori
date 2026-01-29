# Python FastAPI Gleeee

A production-ready FastAPI application deployed via Gleeees.

## Features

- ⚡ FastAPI for high-performance APIs
- 🔒 CORS enabled
- 📝 Auto-generated API documentation (Swagger UI)
- 🐘 PostgreSQL ready (optional)
- 🔴 Redis ready (optional)

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
uvicorn main:app --reload

# Visit http://localhost:8000/docs for API documentation
```

## API Endpoints

- `GET /` - Health check
- `GET /api/health` - Detailed health status
- `GET /docs` - Interactive API documentation

## Environment Variables

Create a `.env` file:

```
DATABASE_URL=postgresql://user:pass@localhost/db
REDIS_URL=redis://localhost:6379
```

## Deployment

This Gleeee is automatically deployed to Vercel. Every push to `main` triggers a new deployment.

## Built with Memori

Created using [Gleeees](https://memorilit.com) - the fastest way to deploy cloud development environments.
