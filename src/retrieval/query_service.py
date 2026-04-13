import logging
from typing import Dict, Any

from src.retrieval.retriever import Retriever
from src.models.llm import LanguageModel

logger = logging.getLogger(__name__)


class QueryService:
    """
    End-to-end RAG query pipeline:
    1. Retrieve relevant chunks
    2. Build grounded prompt
    3. Generate answer using LLM
    4. Return answer with source references
    """

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LanguageModel()

    def query(self, question: str) -> Dict[str, Any]:
        """
        Process a natural language question and return
        a grounded answer with sources.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        logger.info(f"Processing query: {question}")

        retrieved_chunks = self.retriever.retrieve_diverse_context(question)

        answer = self.llm.answer_question(
            question=question,
            context_chunks=retrieved_chunks
        )

        sources = self.retriever.format_sources(retrieved_chunks)

        response = {
            "query": question,
            "answer": answer,
            "sources": sources,
            "num_sources": len(sources),
        }

        logger.info("Query processed successfully")
        return response