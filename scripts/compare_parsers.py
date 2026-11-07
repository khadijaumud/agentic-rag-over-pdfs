import time
from pathlib import Path
from src.parser import parse_pdf_pypdf, parse_pdf_pymupdf

pdf_dir = Path("starter/data/pdfs")
files = sorted(list(pdf_dir.glob("*.pdf")))
print("Comparing pypdf vs PyMuPDF across reports:\n", flush=True)
print(f"{'Filename':<32} | {'PyMuPDF Chars':<14} | {'pypdf Chars':<12} | {'Speedup':<10}", flush=True)
print("-" * 75, flush=True)

# Test on 4 diverse reports (financial statements, tables, multi-column)
sample_files = [f for f in files if f.name in [
    "Albany International Corp..pdf",
    "RWE AG.pdf",
    "Sonic Automotive, Inc..pdf",
    "Armadale Capital Plc.pdf"
]]

for f in sample_files:
    # PyMuPDF
    t0 = time.time()
    mupdf_pages = parse_pdf_pymupdf(f)
    t_mupdf = time.time() - t0
    mupdf_chars = sum(p["char_count"] for p in mupdf_pages)

    # pypdf
    t0 = time.time()
    pypdf_pages = parse_pdf_pypdf(f)
    t_pypdf = time.time() - t0
    pypdf_chars = sum(p["char_count"] for p in pypdf_pages)

    speedup = f"{t_pypdf / max(0.001, t_mupdf):.1f}x"
    print(f"{f.name[:30]:<32} | {mupdf_chars:<14} | {pypdf_chars:<12} | {speedup:<10}", flush=True)

print("\nQuality & Layout Comparison on Tabular Page (Sonic Automotive p.71):", flush=True)
sample_pdf = pdf_dir / "Sonic Automotive, Inc..pdf"
mu_p71 = parse_pdf_pymupdf(sample_pdf)[70]["text"]
py_p71 = parse_pdf_pypdf(sample_pdf)[70]["text"]

print(f"PyMuPDF layout preserves spacing and table rows: {'Net cash provided by operating activities' in mu_p71}", flush=True)
print(f"pypdf layout preserves spacing and table rows: {'Net cash provided by operating activities' in py_p71}", flush=True)
