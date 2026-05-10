"""
spark_fleet/semantic_chunker.py

Selects the most sponsor-relevant text chunks from a PDF using semantic
embeddings (sentence-transformers), replacing simple keyword counting.

How it works
------------
1. Split every PDF page into smaller 300-char overlapping chunks.
2. Embed all chunks with a lightweight local model (all-MiniLM-L6-v2, ~80MB).
3. Embed a fixed "sponsor query" phrase.
4. Rank chunks by cosine similarity to the query.
5. Concatenate the top-ranked chunks until max_chars is reached.

This produces much better input for the LLM than keyword counting because
semantic similarity understands that "Diamond Exhibitor" or "Funding Partner"
are sponsorship signals even without the exact word "sponsor".

Fallback
--------
If `sentence-transformers` is not installed, the module automatically falls
back to the keyword-based approach from run_pipeline.py so the pipeline
never crashes.

Installation
------------
    pip install sentence-transformers

The model is downloaded once on first use and cached locally by
sentence-transformers in %USERPROFILE%\\.cache\\huggingface\\.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

# The query phrase used to find sponsor-relevant chunks.
# Covers all common ways conference brochures reference sponsors.
_SPONSOR_QUERY = (
    "company sponsor exhibitor gold silver platinum bronze diamond "
    "supporter partner funder funding contributing organisation"
)

# Lightweight model — 80MB, fast, runs on CPU.
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

# Chunk size and overlap (in characters)
_CHUNK_SIZE    = 300
_CHUNK_OVERLAP = 50

# Keyword fallback (used if sentence-transformers is not installed)
_SPONSOR_KEYWORDS = [
    "sponsor", "gold", "silver", "platinum", "bronze", "diamond",
    "exhibitor", "partner", "supporter", "funder",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sponsor_text(pages: list, max_chars: int = 5000) -> str:
    """
    Return the most sponsor-relevant text from a list of PDF pages, capped at
    ``max_chars`` characters.

    Tries the semantic (embedding) approach first; falls back to keyword
    scoring if sentence-transformers is not installed.

    Parameters
    ----------
    pages     : List of page objects with a ``.text`` attribute (from pdf_parser).
    max_chars : Maximum characters to return. Default 5000.

    Returns
    -------
    A single string containing the selected chunks, separated by blank lines.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        return _semantic_extract(pages, max_chars, SentenceTransformer)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Falling back to keyword-based text selection. "
            "Run `pip install sentence-transformers` for better results."
        )
        return _keyword_extract(pages, max_chars)


# ---------------------------------------------------------------------------
# Semantic extraction (primary)
# ---------------------------------------------------------------------------

def _chunk_page(text: str) -> list[str]:
    """Split a page's text into overlapping fixed-size chunks."""
    chunks = []
    start  = 0
    while start < len(text):
        end   = start + _CHUNK_SIZE
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += _CHUNK_SIZE - _CHUNK_OVERLAP
    return chunks


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity (avoids numpy dependency)."""
    dot     = sum(x * y for x, y in zip(a, b))
    norm_a  = math.sqrt(sum(x * x for x in a))
    norm_b  = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_extract(pages: list, max_chars: int, SentenceTransformer) -> str:
    """Core semantic extraction using sentence-transformers.
    
    IMPORTANT: The model is forced to load on CPU so it does not compete
    with Ollama for GPU VRAM. This prevents the HTTP 500 crash.
    """
    # 1. Build all chunks from all pages
    all_chunks: list[str] = []
    for page in pages:
        text = page.text.strip()
        if text:
            all_chunks.extend(_chunk_page(text))

    if not all_chunks:
        return ""

    logger.info(
        "Semantic chunker: embedding %d chunks from %d pages...",
        len(all_chunks), len(pages),
    )

    # 2. Load model on CPU (NOT GPU!) to leave VRAM free for Ollama's LLM.
    model = SentenceTransformer(_EMBED_MODEL_NAME, device="cpu")

    # 3. Batch-encode all chunks + query in a single pass (much faster)
    query_emb  = model.encode(_SPONSOR_QUERY).tolist()
    chunk_embs = model.encode(all_chunks).tolist()  # batch encode

    # 4. Score each chunk by similarity to the sponsor query
    scored = sorted(
        zip(chunk_embs, all_chunks),
        key=lambda pair: _cosine_similarity(query_emb, pair[0]),
        reverse=True,
    )

    # 5. Concatenate top chunks until we hit max_chars
    collected: list[str] = []
    total = 0
    for _, chunk in scored:
        if total >= max_chars:
            break
        remaining = max_chars - total
        collected.append(chunk[:remaining])
        total += len(chunk)

    result = "\n\n".join(collected)
    logger.info(
        "Semantic chunker: selected %d chars from top %d chunks (max %d chars).",
        len(result), len(collected), max_chars,
    )
    return result


# ---------------------------------------------------------------------------
# Keyword fallback (used when sentence-transformers is not available)
# ---------------------------------------------------------------------------

def _keyword_extract(pages: list, max_chars: int) -> str:
    """Simple keyword-based page scoring (fallback)."""
    def page_score(text: str) -> int:
        lower = text.lower()
        return sum(lower.count(kw) for kw in _SPONSOR_KEYWORDS)

    scored = sorted(pages, key=lambda p: page_score(p.text), reverse=True)

    collected: list[str] = []
    total = 0
    for page in scored:
        chunk = page.text.strip()
        if not chunk:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        collected.append(chunk[:remaining])
        total += len(chunk)

    result = "\n\n".join(collected)
    logger.info(
        "Keyword chunker: extracted %d chars from %d pages (max %d chars).",
        len(result), len(scored), max_chars,
    )
    return result
