import os
import time
import uuid
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, List

from src.ingestion.parser import PDFParser
from src.ingestion.chunker import DocumentChunker
from src.models.vlm import VisionLanguageModel
from src.retrieval.vectordb import VectorDB

logger = logging.getLogger(__name__)


class IngestService:
    """
    End-to-end ingestion pipeline:
    1. Save uploaded PDF
    2. Parse text, tables, and images
    3. Summarize images using VLM
    4. Chunk all content types
    5. Store chunks in ChromaDB
    """

    def __init__(self):
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        self.parser = PDFParser()
        self.chunker = DocumentChunker()
        self.vlm = VisionLanguageModel()
        self.vectordb = VectorDB()

    def save_upload(self, file_obj, filename: str) -> str:
        """
        Save uploaded PDF file to local storage.

        Args:
            file_obj: file-like object
            filename: original filename

        Returns:
            str: saved file path
        """
        safe_name = filename.replace(" ", "_")
        unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
        file_path = self.upload_dir / unique_name

        with open(file_path, "wb") as out_file:
            shutil.copyfileobj(file_obj, out_file)

        logger.info(f"Saved upload to: {file_path}")
        return str(file_path)

    def summarize_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Convert extracted images into text summaries using VLM.
        Limits image processing to avoid timeout in large PDFs.
        """
        image_summaries: List[Dict[str, Any]] = []
        max_images = int(os.getenv("MAX_IMAGES_PER_DOCUMENT", 2))

        selected_images = images[:max_images]
        logger.info(
            f"Processing {len(selected_images)} images out of {len(images)} extracted images"
        )

        for image in selected_images:
            image_path = image.get("image_path")
            page = image.get("page")
            image_id = image.get("image_id")
            section_title = image.get("section_title", "unknown")
            parser_caption = image.get("caption", "").strip()

            if not image_path:
                continue

            try:
                summary = self.vlm.summarize_image(image_path=image_path)

                if parser_caption:
                    combined_summary = (
                        f"Image summary: {summary}\n"
                        f"Parser caption/context: {parser_caption}"
                    )
                else:
                    combined_summary = summary

                image_summaries.append(
                    {
                        "summary": combined_summary,
                        "page": page,
                        "image_id": image_id,
                        "section_title": section_title,
                    }
                )

            except Exception as e:
                logger.warning(
                    f"Skipping image '{image_id}' due to VLM failure: {str(e)}"
                )

        logger.info(f"Generated {len(image_summaries)} image summaries")
        return image_summaries

    def ingest_pdf(self, file_obj, filename: str) -> Dict[str, Any]:
        """
        Full ingestion pipeline for a single PDF.

        Returns:
            Dict[str, Any]: ingestion summary
        """
        start_time = time.time()

        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")

        saved_path = self.save_upload(file_obj=file_obj, filename=filename)
        logger.info(f"Starting parse for file: {saved_path}")

        parsed = self.parser.parse(saved_path)

        text_blocks = parsed.get("text_blocks", [])
        tables = parsed.get("tables", [])
        images = parsed.get("images", [])

        logger.info(
            f"Parsed content counts - text_blocks: {len(text_blocks)}, "
            f"tables: {len(tables)}, images: {len(images)}"
        )

        image_summaries = self.summarize_images(images=images)

        documents = self.chunker.create_all_chunks(
            text_blocks=text_blocks,
            tables=tables,
            image_summaries=image_summaries,
            filename=parsed["filename"],
        )

        logger.info(f"Created {len(documents)} total chunk documents")

        inserted_count = self.vectordb.add_documents(documents)

        processing_time = round(time.time() - start_time, 2)

        summary = {
            "message": "Document ingested successfully",
            "filename": parsed["filename"],
            "document_path": parsed["document_path"],
            "num_pages": parsed["num_pages"],
            "text_blocks": len(text_blocks),
            "tables": len(tables),
            "image_summaries": len(image_summaries),
            "total_chunks_added": inserted_count,
            "processing_time_seconds": processing_time,
        }

        logger.info(f"Ingestion summary: {summary}")
        return summary