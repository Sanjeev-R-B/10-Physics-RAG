import os
import json
import re
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from llama_parse import LlamaParse

load_dotenv()

# ========================= CONFIG =========================
RAW_PYQ_DIR = Path("pyqs/physics/raw_pdfs")
OUTPUT_DIR = Path("pyqs/physics/parsed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSONL = OUTPUT_DIR / "pyqs_parsed.jsonl"

# LlamaParse setup - optimized for PYQ papers
parser = LlamaParse(
    result_type="markdown",
    num_workers=4,
    verbose=True,
    language="en",
    parsing_instruction="""
    - Extract questions clearly with numbering and marks (e.g., Q.1 [3 marks]).
    - Include year if mentioned (e.g., 2024, 2023, Delhi 2022).
    - Preserve MCQ options (A, B, C, D).
    - Keep assertion-reason, case-based, diagram descriptions.
    - Treat each question as a logical block.
    """,
)

def extract_years_from_text(text: str):
    """Extract and return list of unique years found in text"""
    # Common patterns: 2024, 2023, (2022), [2021], 2020-21, etc.
    patterns = [
        r'\b(20\d{2})\b',           # 2024
        r'\b(19\d{2})\b',           # 1999 (rare but possible)
        r'\(20\d{2}\)',             # (2024)
        r'\[20\d{2}\]',             # [2023]
        r'20\d{2}-\d{2}',           # 2023-24
    ]
    
    years = set()
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Extract base year from 2023-24
            if '-' in match:
                match = match.split('-')[0]
            years.add(match)
    
    return sorted(list(years))

def main():
    pdf_files = sorted([f for f in RAW_PYQ_DIR.glob("*.pdf") if not f.name.startswith(".")])
    
    if not pdf_files:
        print("❌ No PDF files found in pyqs/physics/raw_pdfs/")
        print("Add your PYQ compilations like electricity1.pdf, light1.pdf, etc.")
        return
    
    print(f"Found {len(pdf_files)} PYQ compilation PDF(s):")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")
    print("\nStarting parsing...\n")

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        total_parsed = 0
        for pdf_path in pdf_files:
            print(f"Parsing: {pdf_path.name}")
            try:
                documents = parser.load_data(str(pdf_path))
                print(f"   → Extracted {len(documents)} page(s)/section(s)")

                for i, doc in enumerate(documents):
                    raw_text = doc.text
                    
                    # Extract years
                    years_found = extract_years_from_text(raw_text)
                    primary_year = years_found[0] if years_found else None
                    
                    # Additional hint near marks
                    year_hint_match = re.search(r'(20\d{2}|19\d{2}).*?(marks|\d\s*marks)', raw_text, re.IGNORECASE)
                    year_hint = year_hint_match.group(1) if year_hint_match else None

                    metadata = {
                        "source_file": pdf_path.name,
                        "source_type": "pyq_compilation",
                        "document_index": i,
                        "parser": "llamaparse",
                        "extraction_date": datetime.now().strftime("%Y-%m-%d"),
                        "type": "pyq",
                        "years": years_found,                    # e.g., ["2024", "2023"]
                        "primary_year": primary_year,            # e.g., "2024"
                        "year_hint": year_hint,                  # e.g., "2024"
                        "text_length": len(raw_text)
                    }
                    
                    # Merge with LlamaParse's own metadata (page numbers, etc.)
                    metadata.update(getattr(doc, "metadata", {}))

                    entry = {
                        "id": f"pyq_{pdf_path.stem}_{i}",
                        "text": raw_text,
                        "metadata": metadata
                    }
                    
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    total_parsed += 1
                
                print(f"✓ Completed {pdf_path.name}\n")
                
            except Exception as e:
                print(f"✗ Error parsing {pdf_path.name}: {str(e)}\n")
    
    print(f"🎉 SUCCESS! Parsed {total_parsed} question blocks total")
    print(f"Output saved to: {OUTPUT_JSONL}")
    print("\nNext: Run chunking script → pyqs_chunks.jsonl")
    print("Then: Use years in display like '(2024, 3 marks)'")

if __name__ == "__main__":
    main()