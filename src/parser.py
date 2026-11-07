"""PDF Parser with page-level extraction.
Supports PyMuPDF (fitz) and pypdf with layout awareness.
Preserves exact 1-based page numbers.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
import re

def clean_extracted_text(text: str) -> str:
    """Normalize whitespace and strip excessive repeated blank lines."""
    if not text:
        return ""
    # Replace null bytes
    text = text.replace("\x00", "")
    # Normalize unicode spaces
    text = text.replace("\u00a0", " ")
    # Collapse 3+ newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def parse_pdf_pypdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract pages using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    pages_data = []
    for idx, page in enumerate(reader.pages):
        page_num = idx + 1 # 1-based
        raw_text = page.extract_text() or ""
        clean_text = clean_extracted_text(raw_text)
        pages_data.append({
            "page_number": page_num,
            "text": clean_text,
            "char_count": len(clean_text),
            "parser": "pypdf"
        })
    return pages_data

def parse_pdf_pymupdf(pdf_path: Path) -> List[Dict[str, Any]]:
    """Extract pages using PyMuPDF with sort=True for layout order."""
    import pymupdf
    doc = pymupdf.open(str(pdf_path))
    pages_data = []
    for idx, page in enumerate(doc):
        page_num = idx + 1 # 1-based
        # sort=True sorts text reading order (top-to-bottom, left-to-right)
        raw_text = page.get_text("text", sort=True) or ""
        clean_text = clean_extracted_text(raw_text)
        pages_data.append({
            "page_number": page_num,
            "text": clean_text,
            "char_count": len(clean_text),
            "parser": "pymupdf"
        })
    doc.close()
    return pages_data

def parse_pdf(pdf_path: Path, preferred_parser: str = "auto") -> List[Dict[str, Any]]:
    """Extract all pages from a PDF. Defaults to PyMuPDF if installed, else pypdf."""
    if preferred_parser in ("pymupdf", "fitz", "auto"):
        try:
            return parse_pdf_pymupdf(pdf_path)
        except ImportError:
            pass
        except Exception as e:
            print(f"[Parser Warning] PyMuPDF failed on {pdf_path.name}: {e}. Falling back to pypdf.")
    
    # Fallback to pypdf
    return parse_pdf_pypdf(pdf_path)

def compare_parsers_on_file(pdf_path: Path, sample_pages: Optional[List[int]] = None) -> Dict[str, Any]:
    """Compare extraction quality between PyMuPDF and pypdf on a given PDF."""
    results = {}
    try:
        pypdf_pages = parse_pdf_pypdf(pdf_path)
        results["pypdf_page_count"] = len(pypdf_pages)
        results["pypdf_total_chars"] = sum(p["char_count"] for p in pypdf_pages)
    except Exception as e:
        results["pypdf_error"] = str(e)

    try:
        mupdf_pages = parse_pdf_pymupdf(pdf_path)
        results["pymupdf_page_count"] = len(mupdf_pages)
        results["pymupdf_total_chars"] = sum(p["char_count"] for p in mupdf_pages)
    except Exception as e:
        results["pymupdf_error"] = str(e)
    
    return results
