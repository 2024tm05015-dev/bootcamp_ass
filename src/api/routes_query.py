from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging

from src.retrieval.query_service import QueryService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Query"])

query_service = QueryService()


class QueryRequest(BaseModel):
    query: str


@router.post("/query")
def query_system(request: QueryRequest):
    """
    Query the RAG system:
    - Accepts natural language question
    - Retrieves relevant chunks
    - Generates grounded answer
    - Returns answer with sources
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty.")

        logger.info(f"Incoming query: {request.query}")

        result = query_service.query(request.query)

        return result

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")