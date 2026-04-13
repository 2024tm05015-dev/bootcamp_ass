import os
import logging
from typing import Optional

from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

logger = logging.getLogger(__name__)


class VisionLanguageModel:
    """
    Local image summarization using a Hugging Face BLIP model.

    This avoids remote Hugging Face inference API calls and runs the
    captioning model locally instead.
    """

    def __init__(self):
        self.model_name = os.getenv(
            "VISION_MODEL",
            "Salesforce/blip-image-captioning-base"
        )

        self.max_new_tokens = int(os.getenv("MAX_NEW_TOKENS", 64))

        # Use CPU by default if CUDA is not available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            f"Loading local vision model '{self.model_name}' on device '{self.device}'"
        )

        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        logger.info("Local vision model loaded successfully")

    @staticmethod
    def _read_image(image_path: str) -> Image.Image:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        return Image.open(image_path).convert("RGB")

    def summarize_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """
        Generate a concise summary/caption for an image file.

        Args:
            image_path: Local path to the extracted image
            prompt: Optional guiding prompt

        Returns:
            str: Generated image summary
        """
        try:
            image = self._read_image(image_path)

            logger.info(f"Generating local summary for image: {image_path}")

            if prompt and prompt.strip():
                inputs = self.processor(
                    images=image,
                    text=prompt.strip(),
                    return_tensors="pt"
                )
            else:
                inputs = self.processor(
                    images=image,
                    return_tensors="pt"
                )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens
                )

            caption = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True
            ).strip()

            if not caption:
                raise ValueError("Empty caption returned by local vision model.")

            return caption

        except Exception as e:
            logger.error(f"Image summarization failed for '{image_path}': {str(e)}")
            raise

    def summarize_image_base64(
        self,
        image_path: str,
        prompt: Optional[str] = None
    ) -> str:
        """
        Kept only for interface compatibility.
        Local BLIP flow does not need base64 conversion.
        """
        raise NotImplementedError(
            "Base64 summarization is not used in local BLIP mode."
        )