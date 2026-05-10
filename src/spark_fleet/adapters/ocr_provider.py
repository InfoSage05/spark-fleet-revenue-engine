"""
spark_fleet/adapters/ocr_provider.py

OCR fallback adapter using Tesseract via the `pytesseract` library.
This is used by `pdf_parser.py` when it encounters image-only PDF pages
that do not have a selectable text layer.
"""

from __future__ import annotations

import io
import logging

logger = logging.getLogger(__name__)

class TesseractAdapter:
    """
    Implements the OcrAdapter protocol for PyMuPDF.
    
    Extracts text from a page image using Google's Tesseract OCR engine.
    """

    def __init__(self, tesseract_cmd: str | None = None) -> None:
        """
        Parameters
        ----------
        tesseract_cmd : Optional path to the tesseract executable.
                        On Windows, this is usually r'C:\Program Files\Tesseract-OCR\tesseract.exe'.
                        If None, pytesseract assumes it's in the system PATH.
        """
        self._available = False
        try:
            import pytesseract
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            
            # Simple check to see if the binary is accessible
            pytesseract.get_tesseract_version()
            self._available = True
            logger.info("Tesseract OCR is available and ready for fallback.")
        except ImportError:
            logger.warning("pytesseract or pillow is not installed. OCR will not be available.")
        except Exception as e:
            logger.warning(
                "Tesseract binary not found or inaccessible. OCR fallback disabled.\n"
                "On Windows, download it from https://github.com/UB-Mannheim/tesseract/wiki\n"
                f"Error: {e}"
            )

    def extract_page_text(self, page_image_bytes: bytes) -> str:
        """
        Run OCR on the provided PNG image bytes.
        Returns an empty string if OCR is unavailable or fails.
        """
        if not self._available:
            return ""

        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(page_image_bytes))
            text = pytesseract.image_to_string(img)
            return text.strip()
        except Exception as e:
            logger.error("OCR extraction failed for page: %s", e)
            return ""
