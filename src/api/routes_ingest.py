from io import BytesIO
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from src.ingestion.ingest_service import IngestService

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Ingestion"])

ingest_service = IngestService()
job_store = {}


@router.post("/ingest")
async def ingest_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Start PDF ingestion in the background:
    - Save uploaded file
    - Parse text, tables, images
    - Summarize images (VLM)
    - Chunk all content
    - Store in vector DB

    Returns immediately with a job_id.
    """
    try:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        logger.info(f"Received file for ingestion: {file.filename}")

        file_bytes = await file.read()
        job_id = str(uuid.uuid4())

        job_store[job_id] = {
            "status": "processing",
            "filename": file.filename
        }

        def run_ingestion():
            try:
                result = ingest_service.ingest_pdf(
                    file_obj=BytesIO(file_bytes),
                    filename=file.filename
                )

                job_store[job_id] = {
                    "status": "completed",
                    "filename": file.filename,
                    "result": result
                }

                logger.info(f"Ingestion completed successfully for job_id={job_id}")

            except Exception as e:
                logger.exception(f"Ingestion failed for job_id={job_id}: {str(e)}")
                job_store[job_id] = {
                    "status": "failed",
                    "filename": file.filename,
                    "error": str(e)
                }

        background_tasks.add_task(run_ingestion)

        return JSONResponse(
            content={
                "message": "Ingestion started successfully",
                "job_id": job_id,
                "filename": file.filename,
                "status": "processing"
            }
        )

    except HTTPException as e:
        raise e

    except Exception as e:
        logger.error(f"Failed to start ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start ingestion: {str(e)}")


@router.get("/ingest-status/{job_id}")
def get_ingest_status(job_id: str):
    """
    Check background ingestion status by job_id.
    """
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Invalid job_id")

    return JSONResponse(content=job_store[job_id])