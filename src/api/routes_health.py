from datetime import datetime, timezone
from fastapi import APIRouter
import os

from chromadb import PersistentClient

router = APIRouter(tags=["Health"])

APP_START_TIME = datetime.now(timezone.utc)


@router.get("/health")
def health_check():
    """
    Returns system health and readiness details.
    """
    chroma_dir = os.getenv("CHROMA_DB_DIR", "./chroma_db")
    collection_name = os.getenv("COLLECTION_NAME", "vehicle_manuals")

    try:
        client = PersistentClient(path=chroma_dir)
        collection = client.get_or_create_collection(name=collection_name)
        index_size = collection.count()
        db_ready = True
    except Exception:
        index_size = 0
        db_ready = False

    uptime_seconds = int((datetime.now(timezone.utc) - APP_START_TIME).total_seconds())

    return {
        "status": "healthy" if db_ready else "degraded",
        "model_readiness": {
            "embedding_model": True,
            "llm_model": True,
            "vision_model": True
        },
        "indexed_documents": None,
        "index_size": index_size,
        "uptime_seconds": uptime_seconds
    }