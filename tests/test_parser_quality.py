"""Focused parser quality checks for layout text and image-only PDFs."""

import pytest

from src.parser import parse_pdf


fitz = pytest.importorskip("pymupdf")


def test_table_like_layout_preserves_cell_text_and_reading_order(tmp_path):
    pdf_path = tmp_path / "table.pdf"
    document = fitz.open()
    page = document.new_page()
    page.insert_text((50, 50), "Metric        2024        2023")
    page.insert_text((50, 70), "Revenue       120         100")
    page.insert_text((50, 90), "Employees     40          35")
    document.save(pdf_path)
    document.close()

    pages = parse_pdf(pdf_path)

    assert len(pages) == 1
    assert "Metric" in pages[0]["text"]
    assert "Revenue" in pages[0]["text"]
    assert "Employees" in pages[0]["text"]
    assert pages[0]["text"].index("Revenue") < pages[0]["text"].index("Employees")


def test_image_only_pdf_is_explicitly_unsupported_without_ocr(tmp_path):
    pdf_path = tmp_path / "scanned.pdf"
    image = fitz.Pixmap(fitz.csRGB, (0, 0, 160, 80), False)
    image.clear_with(255)
    document = fitz.open()
    page = document.new_page(width=160, height=80)
    page.insert_image(page.rect, pixmap=image)
    document.save(pdf_path)
    document.close()

    pages = parse_pdf(pdf_path)

    assert len(pages) == 1
    assert pages[0]["text"] == ""
    assert pages[0]["char_count"] == 0
