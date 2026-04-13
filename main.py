import os
import time
import logging

from dotenv import load_dotenv
from pathlib import Path

from fastapi import FastAPI
from dotenv import load_dotenv

# Load .env FIRST, before importing any app modules that use env vars
env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

from src.api.routes_health import router as health_router
from src.api.routes_ingest import router as ingest_router
from src.api.routes_query import router as query_router

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Multimodal RAG System - Vehicle Manual Assistant",
    description="RAG-based system for querying vehicle manuals using text, tables, and images",
    version="1.0.0"
)

start_time = time.time()


@app.on_event("startup")
async def startup_event():
    logger.info("Starting Multimodal RAG API...")


@app.get("/")
async def root():
    return {
        "message": "Multimodal RAG API is running",
        "docs": "/docs"
    }


app.include_router(health_router)
app.include_router(ingest_router)
app.include_router(query_router)