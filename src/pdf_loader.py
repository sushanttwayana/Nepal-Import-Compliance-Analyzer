"""
pdf_loader.py — Extract plain text from PDF files using pdfplumber.

Why pdfplumber?
  - Handles multi-column layouts better than basic PyPDF2
  - Returns page-level text, which we join into one string
  - Gracefully returns empty string if a page has no extractable text
    (e.g., image-only page)
"""

from pathlib import Path
from typing import Optional
import pdfplumber


def extract_text_from_pdf(pdf_path: Path, max_chars: Optional[int] = None) -> str:
    """
    Open a PDF and return all extractable text as a single string.

    Args:
        pdf_path:  Full path to the PDF file.
        max_chars: If set, truncate the output to this many characters.
                   Keeps the prompt within safe token limits.

    Returns:
        Extracted text string.  If extraction fails, returns an error message
        string so the pipeline can continue and report the problem gracefully.
    """
    pages_text: list[str] = []

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    pages_text.append(f"--- Page {i + 1} of {total_pages} ---\n{page_text.strip()}")

        full_text = "\n\n".join(pages_text)

        if not full_text.strip():
            return (
                f"[WARNING] No extractable text found in '{pdf_path.name}'. "
                "The PDF may be image-based (scanned). "
                "Manual transcription or OCR would be needed."
            )

        if max_chars and len(full_text) > max_chars:
            full_text = (
                full_text[:max_chars]
                + f"\n\n[... text truncated at {max_chars} characters to stay within LLM token limits ...]"
            )

        return full_text

    except FileNotFoundError:
        return f"[ERROR] File not found: {pdf_path}"
    except Exception as exc:
        return f"[ERROR] Could not read '{pdf_path.name}': {exc}"
