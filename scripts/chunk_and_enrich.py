import os
import json
import tiktoken
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

load_dotenv()

# ========================= CONFIG =========================
SUBJECT = "physics"
PARSED_JSONL = Path("parsed") / SUBJECT / "parsed.jsonl"
CHUNKS_DIR = Path("chunks") / SUBJECT
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKS_JSONL = CHUNKS_DIR / "chunks.jsonl"

# Optional Grok for keywords/summary
grok_key = os.getenv("GROK_API_KEY")
if grok_key:
    print("Using Grok (xAI) for keywords and summaries...")
    Settings.llm = OpenAILike(
        model="grok-beta",
        api_key=grok_key,
        api_base="https://api.x.ai/v1",
        temperature=0.0,
    )
else:
    print("No GROK_API_KEY found. Skipping keywords/summary.")
    Settings.llm = None

print("Loading local embedding model...")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

tokenizer = tiktoken.get_encoding("cl100k_base")

# ====================== HELPERS ======================
def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

def extract_keywords_and_summary(text: str):
    if Settings.llm is None:
        return [], "Summary not generated (no LLM available)."

    if count_tokens(text) > 3000:
        text = text[:3000] + "... [truncated]"

    prompt = f"""
You are an expert Class 10 Physics teacher. From this textbook paragraph:

Extract:
1. 5–8 most important keywords or short phrases
2. One clear, concise summary sentence

Return ONLY valid JSON:
{{"keywords": ["term1", "term2", ...], "summary": "One sentence."}}

Text:
{text}
"""

    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        result = json.loads(raw)
        return result.get("keywords", [])[:8], result.get("summary", "") or "Summary generated."
    except Exception as e:
        print(f"   ⚠️ Grok failed: {e}")
        return [], "Summary failed."

# ====================== MAIN ======================
def main():
    print(f"Loading parsed chapters from {PARSED_JSONL}...")
    documents = []
    with open(PARSED_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)

            # =====================================================
            # FIX: Preserve the correct source filename from metadata.
            # Previously, all chunks from parsed.jsonl got a wrong or
            # default source name. We now explicitly carry it forward.
            # =====================================================
            metadata = data["metadata"]

            # Make sure "source" is set to the actual PDF filename.
            # parsed.jsonl should have e.g. "source_file": "electricity1.pdf"
            # We normalize to always store it under the key "source".
            source_file = (
                metadata.get("source_file")      # try source_file first
                or metadata.get("source")         # fallback to source
                or metadata.get("file_name")      # fallback to file_name
                or "unknown.pdf"
            )
            # Overwrite so downstream always reads metadata["source"]
            metadata["source"] = source_file

            documents.append(Document(
                text=data["text"],
                metadata=metadata,
                id_=data["id"]
            ))

    if len(documents) == 0:
        print("❌ No documents loaded! Check parsed.jsonl path and content.")
        return

    print(f"Loaded {len(documents)} chapter(s). Starting chunking...\n")

    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=100,
        paragraph_separator="\n\n",
        separator=" ",
    )

    nodes = splitter.get_nodes_from_documents(documents)
    print(f"Generated {len(nodes)} chunks. Enriching metadata...\n")

    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for i, node in enumerate(nodes):
            print(f"  Processing chunk {i+1}/{len(nodes)}", end="\r")

            words = len(node.text.split())
            level = "section" if words > 300 else "subsection" if words > 150 else "paragraph"

            lines = node.text.strip().split("\n")
            section = ""
            subsection = ""
            for line in lines[:5]:
                s = line.strip()
                if s.startswith("### "):
                    subsection = s[4:].strip()
                elif s.startswith("## "):
                    section = s[3:].strip()
                elif s.startswith("# "):
                    section = s[2:].strip()

            chunk_type = "example" if any(k in node.text for k in ["Example", "Activity", "Solved", "Question"]) else "explanation"

            keywords, summary = extract_keywords_and_summary(node.text)

            # =====================================================
            # FIX: source is now correctly inherited from the
            # document metadata (set above), not defaulting to a
            # wrong file. Each chunk carries its actual PDF name.
            # =====================================================
            chunk = {
                "id": node.id_,
                "text": node.text,
                "metadata": {
                    **node.metadata,   # includes the corrected "source"
                    "hierarchy_level": level,
                    "section": section or node.metadata.get("chapter_title", ""),
                    "subsection": subsection,
                    "token_count": count_tokens(node.text),
                    "type": chunk_type,
                },
                "parent_id": None,
                "child_ids": [],
                "previous_id": node.prev_node.node_id if node.prev_node else None,
                "next_id": node.next_node.node_id if node.next_node else None,
                "keywords": keywords,
                "summary": summary
            }
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"\n✅ {len(nodes)} chunks saved to {CHUNKS_JSONL}")

if __name__ == "__main__":
    main()