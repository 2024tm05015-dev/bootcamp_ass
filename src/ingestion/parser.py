import os
import uuid
import logging
from pathlib import Path
from typing import Dict, Any, List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TableFormerMode,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem

logger = logging.getLogger(__name__)


class PDFParser:
    """
    Parse multimodal PDF documents using Docling and extract:
    - text blocks
    - tables
    - images (saved locally for later VLM summarization)
    """

    def __init__(self):
        self.image_output_dir = Path("data/extracted_images")
        self.image_output_dir.mkdir(parents=True, exist_ok=True)

        self.images_scale = float(os.getenv("IMAGE_RESOLUTION_SCALE", "2.0"))

    def _build_converter(self) -> DocumentConverter:
        pipeline_options = PdfPipelineOptions()
        pipeline_options.images_scale = self.images_scale
        pipeline_options.generate_picture_images = True
        pipeline_options.do_table_structure = True
        pipeline_options.do_ocr = False
        pipeline_options.table_structure_options = TableStructureOptions(
            do_cell_matching=True,
            mode=TableFormerMode.ACCURATE,
        )

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )
        return converter

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _extract_text_blocks(self, conv_res) -> List[Dict[str, Any]]:
        """
        Extract full document text as one or more large text blocks.
        This avoids incompatible page filtering calls.
        """
        text_blocks: List[Dict[str, Any]] = []

        try:
            full_text = self._safe_text(conv_res.document.export_to_text())

            if full_text:
                text_blocks.append(
                    {
                        "text": full_text,
                        "page": 1,
                        "section_title": "Full Document",
                    }
                )

        except Exception as e:
            logger.warning(f"Failed to extract document text: {str(e)}")

        logger.info(f"Extracted {len(text_blocks)} text blocks")
        return text_blocks

    def _extract_tables(self, conv_res) -> List[Dict[str, Any]]:
        tables: List[Dict[str, Any]] = []

        for table_idx, table in enumerate(conv_res.document.tables, start=1):
            try:
                df = table.export_to_dataframe(doc=conv_res.document)

                try:
                    content = df.to_markdown(index=False)
                except Exception:
                    content = df.to_csv(index=False)

                page_no = None
                if getattr(table, "prov", None):
                    try:
                        page_no = table.prov[0].page_no
                    except Exception:
                        page_no = None

                tables.append(
                    {
                        "content": self._safe_text(content),
                        "page": page_no,
                        "table_id": f"table_{table_idx}",
                        "section_title": f"Table {table_idx}",
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to extract table {table_idx}: {str(e)}")

        logger.info(f"Extracted {len(tables)} tables")
        return tables

    def _extract_images(self, conv_res, filename_stem: str) -> List[Dict[str, Any]]:
        images: List[Dict[str, Any]] = []
        picture_counter = 0

        for element, _level in conv_res.document.iterate_items():
            if not isinstance(element, PictureItem):
                continue

            picture_counter += 1
            image_id = f"image_{picture_counter}"
            image_filename = f"{filename_stem}_{image_id}_{uuid.uuid4().hex[:8]}.png"
            image_path = self.image_output_dir / image_filename

            try:
                pil_image = element.get_image(conv_res.document)
                pil_image.save(image_path, "PNG")

                page_no = None
                if getattr(element, "prov", None):
                    try:
                        page_no = element.prov[0].page_no
                    except Exception:
                        page_no = None

                caption = ""
                if hasattr(element, "caption_data") and element.caption_data:
                    caption = self._safe_text(element.caption_data)

                images.append(
                    {
                        "image_path": str(image_path),
                        "page": page_no,
                        "image_id": image_id,
                        "section_title": f"Image {picture_counter}",
                        "caption": caption,
                    }
                )

            except Exception as e:
                logger.warning(f"Failed to extract image {image_id}: {str(e)}")

        logger.info(f"Extracted {len(images)} images")
        return images

    def parse(self, pdf_path: str) -> Dict[str, Any]:
        input_path = Path(pdf_path)

        if not input_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if input_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        logger.info(f"Parsing PDF: {input_path}")

        converter = self._build_converter()
        conv_res = converter.convert(input_path)

        text_blocks = self._extract_text_blocks(conv_res)
        tables = self._extract_tables(conv_res)
        images = self._extract_images(conv_res, filename_stem=input_path.stem)

        result = {
            "filename": input_path.name,
            "document_path": str(input_path),
            "num_pages": len(conv_res.document.pages),
            "text_blocks": text_blocks,
            "tables": tables,
            "images": images,
        }

        logger.info(
            "Parsing complete for '%s': text_blocks=%s, tables=%s, images=%s",
            input_path.name,
            len(text_blocks),
            len(tables),
            len(images),
        )

        return result