import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# ========================= CONFIG =========================
CHUNKS_JSONL = Path("pyqs/physics/chunks/pyqs_chunks.jsonl")
INDEX_NAME = "physics-pyqs"

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(INDEX_NAME)

vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

def clean_metadata(metadata: dict) -> dict:
    """Remove None values and ensure Pinecone-compatible types"""
    cleaned = {}
    for k, v in metadata.items():
        if v is None:
            continue  # Skip None values
        if isinstance(v, list):
            # Ensure list contains only strings/numbers
            cleaned[k] = [str(item) if item is not None else "" for item in v]
        else:
            cleaned[k] = str(v) if not isinstance(v, (str, int, float, bool)) else v
    return cleaned

def main():
    print(f"Loading PYQ chunks from {CHUNKS_JSONL}...")
    
    if not CHUNKS_JSONL.exists():
        print("File not found! Run chunking first.")
        return

    nodes = []
    with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            from llama_index.core.schema import TextNode
            
            # Clean metadata before creating node
            cleaned_meta = clean_metadata(chunk["metadata"])
            
            node = TextNode(
                text=chunk["text"],
                id_=chunk["id"],
                metadata=cleaned_meta
            )
            nodes.append(node)

    print(f"Loaded and cleaned {len(nodes)} PYQ chunks.")

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print(f"Indexing into Pinecone: {INDEX_NAME}...")
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print(f"\n🎉 SUCCESS! {len(nodes)} PYQ chunks indexed in '{INDEX_NAME}'")
    print("All metadata cleaned — no more 'null' errors!")
    print("\nYour real board questions are now searchable with years and marks!")

if __name__ == "__main__":
    main()