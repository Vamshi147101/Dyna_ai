# extract_pdfs.py
import fitz  # pymupdf
import os
from pathlib import Path

def extract_all_pdfs(manual_dir, output_dir="extracted_manuals"):
    """
    Extracts text from all PDFs in manual_dir.
    Creates one .txt file per PDF, preserving page numbers.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    manual_dir = Path(manual_dir)
    pdf_files = list(manual_dir.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDFs found in {manual_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDFs")
    
    for pdf_path in pdf_files:
        print(f"Processing: {pdf_path.name}...")
        doc = fitz.open(str(pdf_path))
        
        # Get page count BEFORE we start reading
        num_pages = len(doc)
        
        output_file = Path(output_dir) / f"{pdf_path.stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for page_num in range(num_pages):
                page = doc[page_num]
                text = page.get_text()
                
                # Only write pages with content
                if text.strip():
                    f.write(f"\n--- PAGE {page_num + 1} ---\n")
                    f.write(text)
        
        doc.close()
        print(f"  → {output_file} ({num_pages} pages)")
    
    print(f"\nDone. Extracted texts in: {output_dir}/")

# Run it
extract_all_pdfs("D:\\vamshi\\ML\\K-files\\Dyna_ai\\pdfs")