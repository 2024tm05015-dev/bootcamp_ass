import os
import logging
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for Hugging Face embedding model used across
    ingestion and retrieval pipelines.
    """

    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
        self._model = None

    def load(self) -> HuggingFaceEmbeddings:
        """
        Load and return the embedding model instance.
        Reuses the model if already loaded.
        """
        if self._model is None:
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = HuggingFaceEmbeddings(
                model_name=self.model_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info("Embedding model loaded successfully")
        return self._model

    @property
    def model(self) -> HuggingFaceEmbeddings:
        """
        Property accessor for the loaded embedding model.
        """
        return self.load()

    def embed_query(self, query: str) -> list[float]:
        """
        Generate embedding for a user query.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")
        return self.model.embed_query(query)

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of text documents.
        """
        if not documents:
            raise ValueError("Document list cannot be empty.")
        return self.model.embed_documents(documents)