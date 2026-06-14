import os
import json
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ========================= CONFIG =========================
SUBJECT = "physics"
CHUNKS_JSONL = Path("chunks") / SUBJECT / "chunks.jsonl"
INDEX_DIR = Path("indexes") / SUBJECT
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Pinecone setup
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "physics-textbook")

# Create index if not exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {INDEX_NAME}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # bge-small-en-v1.5
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        )
    )
    print("Index created!")

# Vector store
vector_store = PineconeVectorStore(pinecone_index=pc.Index(INDEX_NAME))

# Local embeddings (free & fast)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# === OLLAMA LLM SETUP (FREE & LOCAL) ===
print("Setting up local Ollama LLM for answers...")
Settings.llm = Ollama(
    model="llama3.2:1b",        # ← New small model
    request_timeout=120.0,
    temperature=0.0
)
Settings.embed_model = embed_model

# ====================== BUILD OR LOAD INDEX ======================
def build_or_load_index():
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    stats = pc.Index(INDEX_NAME).describe_index_stats()
    if stats.total_vector_count > 0:
        print(f"Loading existing index with {stats.total_vector_count} vectors...")
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model,
        )
    else:
        print("Building new index from chunks...")
        nodes = []
        with open(CHUNKS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                from llama_index.core.schema import TextNode
                node = TextNode(
                    text=chunk["text"],
                    id_=chunk["id"],
                    metadata=chunk["metadata"]
                )
                nodes.append(node)

        print(f"Loaded {len(nodes)} chunks. Indexing into Pinecone...")
        index = VectorStoreIndex(
            nodes,
            storage_context=storage_context,
            embed_model=embed_model,
            show_progress=True
        )
        print("Indexing complete!")

    return index

# ====================== CHAT LOOP ======================
def main():
    print("Building/loading Physics textbook index on Pinecone...\n")
    index = build_or_load_index()

    # Full query engine with Ollama synthesis
    query_engine = index.as_query_engine(
        similarity_top_k=2,  # Retrieve top 5 relevant chunks
    )

    print("\n" + "="*70)
    print("   CLASS 10 PHYSICS TEXTBOOK ASSISTANT IS READY!")
    print("   Powered by your textbook + Ollama (local & free)")
    print("="*70)
    print("Ask any question (type 'quit' to exit)\n")

    while True:
        try:
            query = input("Your question: ").strip()
            if query.lower() in ["quit", "exit", "bye"]:
                print("Goodbye! Study well!")
                break
            if not query:
                continue

            print("\nSearching textbook...\n")
            response = query_engine.query(query)

            print("Answer from textbook:\n")
            print(response.response.strip())
            print("\n" + "─"*60)
            print("Sources:")
            for i, source in enumerate(response.source_nodes, 1):
                meta = source.metadata
                section = meta.get("section", "Unknown section")
                source_file = meta.get("source", "unknown")
                print(f"   [{i}] {section} — {source_file}")
            print("\n")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()