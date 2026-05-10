"""
spark_fleet/pdf_parser.py

PyMuPDF-backed PDF text extractor for the Micro Spark (Mac Mini M2).

This module is the bridge between a raw conference brochure PDF and the
text chunks that are sent to the Macro Spark (DGX) for LLM extraction.

Architecture
------------
PDF bytes / path
    │
    ▼
extract_text_from_bytes() / extract_text_from_path()
    │   uses fitz (PyMuPDF) — optional runtime dependency
    │   falls back to OcrAdapter per page if text layer is empty
    ▼
list[PageText]
    │
    ▼
macro_client.build_extraction_prompt()   ← next stage

Installation (Mac Mini M2)
--------------------------
    pip install PyMuPDF

PyMuPDF ships pre-compiled wheels for macOS arm64, so no build toolchain
is required on the Mac Mini.

OCR fallback
------------
Scanned PDFs (image-only pages) have no text layer.  Pass an OcrAdapter
implementation to extract text from those pages via an OCR engine such as
Tesseract.  The adapter interface is a simple Protocol — plug in any
engine without changing this module.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


# ===========================================================================
# Exceptions
# ===========================================================================

class MissingPdfDependency(RuntimeError):
    """
    Raised when PyMuPDF (fitz) is not installed.

    Install it with:  pip install PyMuPDF
    """


# ===========================================================================
# Data containers
# ===========================================================================

@dataclass(frozen=True)
class PageText:
    """
    Text content of a single PDF page.

    Attributes
    ----------
    page     : 1-indexed page number.
    text     : Extracted text, stripped of leading/trailing whitespace.
               Empty string for blank or image-only pages (without OCR).
    used_ocr : True if the text was obtained via an OCR adapter rather than
               the native PDF text layer.
    """
    page:     int
    text:     str
    used_ocr: bool = False


# ===========================================================================
# OCR adapter protocol
# ===========================================================================

@runtime_checkable
class OcrAdapter(Protocol):
    """
    Plug-in interface for an OCR engine.

    Implement this protocol and pass an instance to ``extract_text_from_bytes``
    or ``extract_text_from_path`` to enable OCR on blank pages.

    Example (Tesseract via pytesseract)
    ------------------------------------
    ::

        class TesseractAdapter:
            def extract_page_text(self, page_image_bytes: bytes) -> str:
                import pytesseract
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(page_image_bytes))
                return pytesseract.image_to_string(img)
    """

    def extract_page_text(self, page_image_bytes: bytes) -> str:
        """
        Parameters
        ----------
        page_image_bytes : PNG/JPEG bytes of the rendered page image.

        Returns
        -------
        Extracted text string (may be empty if OCR finds nothing).
        """
        ...


# ===========================================================================
# Core extraction functions
# ===========================================================================

def _import_fitz():
    """
    Lazy-import fitz (PyMuPDF) and surface a readable error when missing.
    Separating the import lets tests patch builtins.__import__ cleanly.
    """
    try:
        import fitz  # type: ignore[import-untyped]
        return fitz
    except ModuleNotFoundError as exc:
        raise MissingPdfDependency(
            "PyMuPDF is required to parse PDFs. "
            "Install it with: pip install PyMuPDF"
        ) from exc


def extract_text_from_bytes(
    pdf_bytes: bytes,
    ocr_adapter: OcrAdapter | None = None,
) -> list[PageText]:
    """
    Extract per-page text from raw PDF bytes.

    Parameters
    ----------
    pdf_bytes   : Raw bytes of the PDF file (e.g. from ``open(path, "rb").read()``).
    ocr_adapter : Optional OCR adapter.  When a page has no native text layer,
                  the adapter is called with the page rendered as a PNG image.

    Returns
    -------
    list[PageText]
        One entry per page, 1-indexed, in document order.
        Pages with no text (and no OCR adapter) have ``text = ""``.

    Raises
    ------
    MissingPdfDependency
        If PyMuPDF is not installed.
    """
    fitz = _import_fitz()
    pages: list[PageText] = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            used_ocr = False

            if not text and ocr_adapter is not None:
                # Render the page to PNG bytes and hand to the OCR adapter.
                mat   = fitz.Matrix(2.0, 2.0)          # 2× scale for accuracy
                pix   = page.get_pixmap(matrix=mat)
                image_bytes = pix.tobytes("png")
                text     = ocr_adapter.extract_page_text(image_bytes).strip()
                used_ocr = True

            pages.append(PageText(page=index, text=text, used_ocr=used_ocr))

    return pages


def extract_text_from_path(
    pdf_path: str | Path,
    ocr_adapter: OcrAdapter | None = None,
) -> list[PageText]:
    """
    Extract per-page text from a PDF file on disk.

    Convenience wrapper around ``extract_text_from_bytes`` — reads the file
    into memory and delegates.

    Parameters
    ----------
    pdf_path    : Absolute or relative path to the PDF file.
    ocr_adapter : Optional OCR adapter (see ``extract_text_from_bytes``).

    Returns
    -------
    list[PageText]

    Raises
    ------
    MissingPdfDependency
        If PyMuPDF is not installed.
    FileNotFoundError
        If the path does not exist.
    """
    path = Path(pdf_path)
    pdf_bytes = path.read_bytes()
    return extract_text_from_bytes(pdf_bytes, ocr_adapter)
