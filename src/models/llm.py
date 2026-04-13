import os
import logging
from typing import List, Dict, Any, Optional

import httpx

from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

logger = logging.getLogger(__name__)


class LanguageModel:
    """
    Wrapper for text generation using OpenRouter Chat Completions API.
    """

    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY", "")
        self.model_name = os.getenv("OPENROUTER_MODEL", "google/gemma-3-27b-it")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_tokens = int(os.getenv("MAX_NEW_TOKENS", 512))
        self.temperature = float(os.getenv("TEMPERATURE", 0.2))

        logger.info(f"OpenRouter API key loaded: {bool(self.api_key)}")
        logger.info(f"Using OpenRouter model: {self.model_name}")

    def _build_headers(self) -> dict:
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is missing in environment variables.")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        app_url = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        app_name = os.getenv("OPENROUTER_APP_NAME", "").strip()

        if app_url:
            headers["HTTP-Referer"] = app_url
        if app_name:
            headers["X-Title"] = app_name

        return headers

    def build_prompt(self, question: str, context_chunks: List[Dict[str, Any]]) -> str:
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not context_chunks:
            return (
                "You are a helpful assistant for vehicle manuals.\n\n"
                f"Question: {question}\n\n"
                "No relevant context was retrieved. Respond honestly that the answer "
                "could not be found in the indexed documents."
            )

        formatted_context_parts = []

        for idx, chunk in enumerate(context_chunks, start=1):
            content = chunk.get("content", "").strip()
            metadata = chunk.get("metadata", {})

            source = (
                f"[Source {idx}] "
                f"filename={metadata.get('filename', 'unknown')}, "
                f"page={metadata.get('page', 'unknown')}, "
                f"chunk_type={metadata.get('chunk_type', 'unknown')}, "
                f"section_title={metadata.get('section_title', 'unknown')}"
            )

            formatted_context_parts.append(f"{source}\n{content}")

        context_text = "\n\n".join(formatted_context_parts)

        prompt = f"""
You are an intelligent assistant for multimodal vehicle manuals.

Answer the user's question only using the retrieved context below.
Follow these rules carefully:
1. Give a clear and concise answer.
2. Use only the provided context. Do not invent facts.
3. If the answer is not fully available in the context, say so clearly.
4. If useful, summarize steps in bullet points.
5. Pay special attention to safety-related instructions and warnings.
6. At the end, include a short "Sources Used" section mentioning filename, page, and chunk type.

Retrieved Context:
{context_text}

User Question:
{question}

Final Answer:
""".strip()

        return prompt

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You answer vehicle-manual questions using only retrieved context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_new_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        try:
            logger.info(f"Generating answer using OpenRouter model: {self.model_name}")

            response = httpx.post(
                self.api_url,
                headers=self._build_headers(),
                json=payload,
                timeout=120.0
            )
            response.raise_for_status()
            result = response.json()

            choices = result.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                if isinstance(content, str) and content.strip():
                    return content.strip()

            error_data = result.get("error")
            if error_data:
                raise ValueError(f"OpenRouter error: {error_data}")

            raise ValueError(f"Unexpected response format from OpenRouter: {result}")

        except Exception as e:
            logger.error(f"LLM generation failed: {str(e)}")
            raise

    def answer_question(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        prompt = self.build_prompt(question=question, context_chunks=context_chunks)
        return self.generate(prompt=prompt)