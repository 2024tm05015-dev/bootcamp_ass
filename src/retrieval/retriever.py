import os
import logging
from typing import List, Dict, Any

from src.retrieval.vectordb import VectorDB

logger = logging.getLogger(__name__)


class Retriever:
    """
    Retrieves relevant chunks from the vector database
    and formats them for downstream LLM consumption.
    """

    def __init__(self):
        self.vectordb = VectorDB()
        self.top_k = int(os.getenv("TOP_K_RESULTS", 6))

    def retrieve(self, query: str, k: int | None = None) -> List[Dict[str, Any]]:
        """
        Retrieve top-k relevant chunks with scores.

        Returns a list of dictionaries:
        {
            "content": "...",
            "metadata": {...},
            "score": 0.123
        }
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        top_k = k or self.top_k
        logger.info(f"Retrieving top {top_k} chunks for query: {query}")

        results = self.vectordb.similarity_search_with_score(query=query, k=top_k)

        formatted_results: List[Dict[str, Any]] = []
        for doc, score in results:
            formatted_results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score),
                }
            )

        logger.info(f"Retrieved {len(formatted_results)} chunks")
        return formatted_results

    def retrieve_diverse_context(
        self,
        query: str,
        k: int | None = None,
        max_per_type: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Retrieve results while maintaining chunk-type diversity.
        Useful for multimodal manuals where text, tables, and image summaries
        may all contribute to the final answer.
        """
        raw_results = self.retrieve(query=query, k=(k or self.top_k) * 2)

        selected: List[Dict[str, Any]] = []
        type_counts: Dict[str, int] = {}

        for item in raw_results:
            chunk_type = item["metadata"].get("chunk_type", "unknown")

            if type_counts.get(chunk_type, 0) < max_per_type:
                selected.append(item)
                type_counts[chunk_type] = type_counts.get(chunk_type, 0) + 1

            if len(selected) >= (k or self.top_k):
                break

        logger.info(
            "Selected %s diverse chunks with type distribution: %s",
            len(selected),
            type_counts
        )

        return selected

    @staticmethod
    def format_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source references for API response.
        """
        sources: List[Dict[str, Any]] = []

        for item in chunks:
            metadata = item.get("metadata", {})
            sources.append(
                {
                    "filename": metadata.get("filename"),
                    "page": metadata.get("page"),
                    "chunk_type": metadata.get("chunk_type"),
                    "section_title": metadata.get("section_title"),
                }
            )

        return sources