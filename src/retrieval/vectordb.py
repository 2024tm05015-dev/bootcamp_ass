import os
import logging
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.models.embedding_model import EmbeddingModel

logger = logging.getLogger(__name__)


class VectorDB:
    """
    Persistent ChromaDB wrapper for storing and retrieving
    multimodal RAG chunks with metadata.
    """

    def __init__(self):
        self.persist_directory = os.getenv("CHROMA_DB_DIR", "./chroma_db")
        self.collection_name = os.getenv("COLLECTION_NAME", "vehicle_manuals")
        self.embedding_model = EmbeddingModel().load()
        self._db: Optional[Chroma] = None

    def load(self) -> Chroma:
        """
        Load or initialize the persistent Chroma collection.
        """
        if self._db is None:
            logger.info(
                f"Loading ChromaDB collection='{self.collection_name}' "
                f"from '{self.persist_directory}'"
            )

            os.makedirs(self.persist_directory, exist_ok=True)

            self._db = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
                persist_directory=self.persist_directory,
            )

            logger.info("ChromaDB loaded successfully")

        return self._db

    @property
    def db(self) -> Chroma:
        """
        Property accessor for the loaded Chroma instance.
        """
        return self.load()

    def add_documents(self, documents: List[Document]) -> int:
        """
        Add LangChain Document objects into the vector store.

        Returns:
            int: Number of documents added
        """
        if not documents:
            raise ValueError("No documents provided for insertion.")

        logger.info(f"Adding {len(documents)} documents to vector store")
        self.db.add_documents(documents)
        logger.info("Documents added successfully")

        return len(documents)

    def similarity_search(
        self,
        query: str,
        k: int = 6,
        filter: Optional[dict] = None
    ) -> List[Document]:
        """
        Retrieve top-k similar documents for the given query.

        Args:
            query: User question
            k: Number of results to retrieve
            filter: Optional metadata filter

        Returns:
            List[Document]: Retrieved documents
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Running similarity search with k={k}")
        results = self.db.similarity_search(query=query, k=k, filter=filter)
        logger.info(f"Retrieved {len(results)} documents")

        return results

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 6,
        filter: Optional[dict] = None
    ):
        """
        Retrieve top-k similar documents with similarity scores.
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        logger.info(f"Running similarity search with scores, k={k}")
        results = self.db.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        logger.info(f"Retrieved {len(results)} scored documents")

        return results

    def count(self) -> int:
        """
        Return the total number of indexed chunks in the collection.
        """
        collection = self.db._collection
        return collection.count()

    def reset(self) -> None:
        """
        Delete all chunks from the current collection.
        Useful during development/testing.
        """
        logger.warning(f"Resetting Chroma collection: {self.collection_name}")
        self.db.delete_collection()
        self._db = None
        self.load()
        logger.info("Collection reset complete")