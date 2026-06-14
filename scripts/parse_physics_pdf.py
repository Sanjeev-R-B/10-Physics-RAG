"""
fix_parsed_jsonl.py
-------------------
Run once to fix duplicate IDs in parsed/physics/parsed.jsonl.
Place this file in your scripts/ folder and run:
    python scripts/fix_parsed_jsonl.py
"""

import json
import uuid
from pathlib import Path

PARSED_JSONL = Path("parsed") / "physics" / "parsed.jsonl"
BACKUP_JSONL = Path("parsed") / "physics" / "parsed_backup.jsonl"

# Chapter name → correct source PDF mapping
# Update this dict to match your actual filenames in raw_pdfs/physics/
CHAPTER_TO_SOURCE = {
    "electricity":   "electricity1.pdf",
    "environment":   "environment1.pdf",
    "human eye":     "humaneye1.pdf",
    "eye":           "humaneye1.pdf",
    "light":         "light1.pdf",
    "reflection":    "light1.pdf",
    "magnetic":      "magnetic1.pdf",
    "sources":       "sources_of_energy.pdf",
}

def infer_source(text: str, current_source: str) -> str:
    """Infer correct source PDF from chapter heading in text."""
    text_lower = text.lower()[:300]  # Check only first 300 chars (heading area)
    for keyword, pdf_name in CHAPTER_TO_SOURCE.items():
        if keyword in text_lower:
            return pdf_name
    return current_source  # Keep existing if no match

def make_unique_id(text: str, index: int) -> str:
    """Generate a stable unique ID from content hash + index."""
    # Use first 60 chars of text + index for uniqueness
    slug = text[:60].strip().lower()
    slug = "".join(c if c.isalnum() else "_" for c in slug)
    return f"physics_ch{index:04d}_{slug[:30]}"

def main():
    if not PARSED_JSONL.exists():
        print(f"❌ File not found: {PARSED_JSONL}")
        return

    # Backup original
    import shutil
    shutil.copy(PARSED_JSONL, BACKUP_JSONL)
    print(f"✅ Backup saved to {BACKUP_JSONL}")

    records = []
    seen_ids = {}

    with open(PARSED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records. Fixing IDs and sources...\n")

    fixed_records = []
    for i, rec in enumerate(records):
        original_id = rec.get("id", "")
        text = rec.get("text", "")
        metadata = rec.get("metadata", {})

        # Fix 1: Generate unique ID
        new_id = make_unique_id(text, i)
        if new_id in seen_ids:
            new_id = f"{new_id}_{uuid.uuid4().hex[:6]}"
        seen_ids[new_id] = True

        # Fix 2: Infer correct source PDF from chapter text
        current_source = metadata.get("source", "unknown.pdf")
        corrected_source = infer_source(text, current_source)

        # Fix 3: Infer chapter_number from text heading
        chapter_number = metadata.get("chapter_number", "unknown")
        if chapter_number == "unknown":
            for line in text.split("\n")[:5]:
                if "CHAPTER" in line.upper():
                    parts = line.upper().replace("CHAPTER", "").strip().split()
                    if parts:
                        chapter_number = parts[0]
                    break

        rec["id"] = new_id
        metadata["source"] = corrected_source
        metadata["chapter_number"] = chapter_number
        rec["metadata"] = metadata

        status = "✅" if corrected_source != current_source else "  "
        print(f"  [{i+1}] {status} ID: {new_id[:50]}")
        print(f"        Source: {current_source} → {corrected_source}")

        fixed_records.append(rec)

    # Write fixed file
    with open(PARSED_JSONL, "w", encoding="utf-8") as f:
        for rec in fixed_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Fixed {len(fixed_records)} records saved to {PARSED_JSONL}")
    print("\nNext steps:")
    print("  1. python scripts/chunk_and_enrich.py")
    print("  2. Delete old Pinecone index vectors")
    print("  3. python scripts/build_index.py")

if __name__ == "__main__":
    main()