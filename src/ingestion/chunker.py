import os
import uuid
import logging
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Creates separate chunk types for:
    - text
    - tables
    - image summaries

    Each chunk is converted into a LangChain Document with rich metadata
    for traceability during retrieval.
    """

    def __init__(self):
        self.chunk_size = int(os.getenv("CHUNK_SIZE", 600))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk_text_blocks(
        self,
        text_blocks: List[Dict[str, Any]],
        filename: str
    ) -> List[Document]:
        """
        Chunk extracted text blocks into smaller pieces.

        Expected input format for each text block:
        {
            "text": "...",
            "page": 1,
            "section_title": "Safety"
        }
        """
        documents: List[Document] = []

        for block in text_blocks:
            text = (block.get("text") or "").strip()
            page = block.get("page")
            section_title = block.get("section_title", "unknown")

            if not text:
                continue

            split_chunks = self.text_splitter.split_text(text)

            for chunk_index, chunk in enumerate(split_chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "chunk_id": str(uuid.uuid4()),
                            "filename": filename,
                            "page": page,
                            "section_title": section_title,
                            "chunk_type": "text",
                            "chunk_index": chunk_index,
                        },
                    )
                )

        logger.info(f"Created {len(documents)} text chunks from '{filename}'")
        return documents

    def chunk_tables(
        self,
        tables: List[Dict[str, Any]],
        filename: str
    ) -> List[Document]:
        """
        Convert extracted tables into retrievable text chunks.

        Expected input format for each table:
        {
            "content": "<markdown or plain text table>",
            "page": 3,
            "table_id": "table_1",
            "section_title": "Child Restraint System"
        }
        """
        documents: List[Document] = []

        for idx, table in enumerate(tables):
            content = (table.get("content") or "").strip()
            page = table.get("page")
            table_id = table.get("table_id", f"table_{idx}")
            section_title = table.get("section_title", "unknown")

            if not content:
                continue

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "chunk_id": str(uuid.uuid4()),
                        "filename": filename,
                        "page": page,
                        "section_title": section_title,
                        "chunk_type": "table",
                        "table_id": table_id,
                        "chunk_index": 0,
                    },
                )
            )

        logger.info(f"Created {len(documents)} table chunks from '{filename}'")
        return documents

    def chunk_image_summaries(
        self,
        image_summaries: List[Dict[str, Any]],
        filename: str
    ) -> List[Document]:
        """
        Convert VLM-generated image summaries into retrievable chunks.

        Expected input format for each image summary:
        {
            "summary": "This image shows ...",
            "page": 5,
            "image_id": "image_2",
            "section_title": "Airbags"
        }
        """
        documents: List[Document] = []

        for idx, image in enumerate(image_summaries):
            summary = (image.get("summary") or "").strip()
            page = image.get("page")
            image_id = image.get("image_id", f"image_{idx}")
            section_title = image.get("section_title", "unknown")

            if not summary:
                continue

            documents.append(
                Document(
                    page_content=summary,
                    metadata={
                        "chunk_id": str(uuid.uuid4()),
                        "filename": filename,
                        "page": page,
                        "section_title": section_title,
                        "chunk_type": "image",
                        "image_id": image_id,
                        "chunk_index": 0,
                    },
                )
            )

        logger.info(f"Created {len(documents)} image-summary chunks from '{filename}'")
        return documents

    def create_all_chunks(
        self,
        text_blocks: List[Dict[str, Any]],
        tables: List[Dict[str, Any]],
        image_summaries: List[Dict[str, Any]],
        filename: str
    ) -> List[Document]:
        """
        Create all chunk types and return a single combined list.
        """
        text_docs = self.chunk_text_blocks(text_blocks=text_blocks, filename=filename)
        table_docs = self.chunk_tables(tables=tables, filename=filename)
        image_docs = self.chunk_image_summaries(
            image_summaries=image_summaries,
            filename=filename
        )

        all_docs = text_docs + table_docs + image_docs

        logger.info(
            f"Total chunks created for '{filename}': "
            f"text={len(text_docs)}, tables={len(table_docs)}, images={len(image_docs)}, total={len(all_docs)}"
        )

        return all_docs