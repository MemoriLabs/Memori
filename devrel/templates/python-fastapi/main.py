from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os

app = FastAPI(
    title="Warplet FastAPI",
    description="A FastAPI application deployed via Warplets",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Warplet FastAPI",
        "timestamp": datetime.now().isoformat(),
        "message": "Welcome to your FastAPI Warplet! 🚀"
    }

@app.get("/api/health")
async def health():
    """Detailed health check"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "environment": os.getenv("ENVIRONMENT", "production")
    }

@app.get("/api/info")
async def info():
    """Get application information"""
    return {
        "name": "Warplet FastAPI",
        "framework": "FastAPI",
        "python_version": "3.11",
        "deployed_with": "Warplets",
        "features": [
            "Fast async API",
            "Auto-generated docs",
            "CORS enabled",
            "Production ready"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
