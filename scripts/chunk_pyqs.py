import json
import tiktoken
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

# ========================= CONFIG =========================
PARSED_JSONL = Path("pyqs/physics/parsed/pyqs_parsed.jsonl")
CHUNKS_DIR = Path("pyqs/physics/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_JSONL = CHUNKS_DIR / "pyqs_chunks.jsonl"

# Tokenizer for accurate token counting
tokenizer = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def main():
    print(f"Loading parsed PYQs from {PARSED_JSONL}...")
    
    if not PARSED_JSONL.exists():
        print(f"❌ File not found: {PARSED_JSONL}")
        print("Run 03_parse_pyqs.py first!")
        return

    documents = []
    with open(PARSED_JSONL, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                doc = Document(
                    text=data["text"],
                    metadata=data["metadata"],
                    id_=data["id"]
                )
                documents.append(doc)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON on line {line_num}: {e}")

    print(f"Loaded {len(documents)} parsed PYQ blocks.\nStarting chunking...\n")

    # Clean splitting optimized for PYQs
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=100,
        paragraph_separator="\n\n",
        separator=" ",
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Generated {len(nodes)} PYQ chunks.\nSaving with full metadata...\n")

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for i, node in enumerate(nodes):
            print(f"  Saving chunk {i+1}/{len(nodes)}", end="\r")

            # Estimate level
            words = len(node.text.split())
            level = "question_group" if words > 200 else "single_question"

            # Fallback year extraction if missing
            meta = node.metadata.copy()
            if "years" not in meta or not meta["years"]:
                import re
                years = re.findall(r'\b(20\d{2}|19\d{2})\b', node.text)
                if years:
                    meta["years"] = sorted(list(set(years)))
                    meta["primary_year"] = meta["years"][0]

            chunk = {
                "id": node.id_,
                "text": node.text.strip(),
                "metadata": {
                    **meta,
                    "chunk_type": "pyq",
                    "hierarchy_level": level,
                    "token_count": count_tokens(node.text),
                    "chunk_index": i
                },
                "previous_id": node.prev_node.node_id if node.prev_node else None,  # FIXED
                "next_id": node.next_node.node_id if node.next_node else None,      # FIXED
            }
            
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n🎉 SUCCESS! {len(nodes)} PYQ chunks created!")
    print(f"Output: {CHUNKS_JSONL}")
    print("\nYour PYQ chunks now include:")
    print("   • Year information (2024, 2023, etc.)")
    print("   • Source file")
    print("   • Marks hints")
    print("   • Ready for display or indexing")

if __name__ == "__main__":
    main()