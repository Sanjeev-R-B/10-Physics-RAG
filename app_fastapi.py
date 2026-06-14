from fastapi import FastAPI, Query
from functools import lru_cache
import os
import json
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

app = FastAPI(
    title="Class 10 Physics Learning Assistant",
    description="Ask questions from textbook OR generate hybrid practice questions (Real PYQs + Original)",
    version="1.0"
)

# ========================= SETUP =========================
print("Setting up LLM and embeddings...")

# FIX 1: Use a larger model for better answers
# llama3.2:1b is too small — it gives 1-line vague answers
# Switch to llama3.2:3b or llama3.1:8b if your system supports it
Settings.llm = Ollama(
    model="llama3.2:3b",        # ← Upgrade from 1b to 3b (better quality)
    request_timeout=180.0,
    temperature=0.1              # ← Lower temp for factual, accurate answers
)
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

# Pinecone connection
print("Connecting to Pinecone...")
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# ========================= FIX 2: CUSTOM QA PROMPT =========================
# Default LlamaIndex prompt is too generic — it gives shallow 1-line answers.
# This custom prompt forces the LLM to give detailed, structured answers.

PHYSICS_QA_PROMPT = PromptTemplate(
    """You are an expert Class 10 Physics teacher helping a student prepare for board exams.
Use ONLY the context provided below to answer the question. Do NOT use outside knowledge.

Instructions:
- Give a clear, detailed explanation (not just one line)
- Include the formula with each symbol explained (if applicable)
- Explain step by step how it works
- Give a real-world example or application if possible
- If the context does not contain enough information to answer, say: "This topic is not covered in the provided textbook sections."

Context from textbook:
---------------------
{context_str}
---------------------

Student's Question: {query_str}

Detailed Answer:"""
)

# ========================= FIX 3: CUSTOM RETRIEVER WITH SCORE FILTER =========================
# similarity_top_k=3 fetches chunks regardless of relevance score.
# We fetch more (top 6) then filter by score to drop irrelevant ones.

class FilteredQueryEngine:
    def __init__(self, index, score_threshold=0.5, top_k=6):
        self.retriever = index.as_retriever(similarity_top_k=top_k)
        self.index = index
        self.score_threshold = score_threshold
        self.top_k = top_k

    def query(self, query_str: str):
        # Step 1: Retrieve candidates
        nodes = self.retriever.retrieve(query_str)

        # Step 2: Filter by relevance score — drop unrelated chunks
        relevant_nodes = [
            n for n in nodes
            if n.score is not None and n.score >= self.score_threshold
        ]

        # Step 3: If all filtered out, fall back to top 2 (avoid empty context)
        if not relevant_nodes:
            relevant_nodes = nodes[:2]

        # Step 4: Build context from relevant chunks only
        context = "\n\n---\n\n".join([n.node.get_content() for n in relevant_nodes])

        # Step 5: Fill custom prompt and call LLM
        prompt = PHYSICS_QA_PROMPT.format(
            context_str=context,
            query_str=query_str
        )
        llm_response = Settings.llm.complete(prompt)

        return {
            "answer": llm_response.text.strip(),
            "source_nodes": relevant_nodes
        }


# Build index and custom query engine
textbook_vector_store = PineconeVectorStore(pinecone_index=pc.Index("physics-textbook"))
textbook_index = VectorStoreIndex.from_vector_store(vector_store=textbook_vector_store)
filtered_engine = FilteredQueryEngine(
    index=textbook_index,
    score_threshold=0.62,  # ← only keeps genuinely relevant chunks
    top_k=6
)

# PYQ index
pyq_vector_store = PineconeVectorStore(pinecone_index=pc.Index("physics-pyqs"))
pyq_index = VectorStoreIndex.from_vector_store(vector_store=pyq_vector_store)
pyq_retriever = pyq_index.as_retriever(similarity_top_k=10)


# ========================= ASK PHYSICS =========================
@lru_cache(maxsize=256)
def get_cached_answer(query: str):
    result = filtered_engine.query(query)

    sources = [
        {
            "section": node.node.metadata.get("section", "Unknown"),
            "source_file": node.node.metadata.get("source", "unknown.pdf"),
            "relevance_score": round(node.score, 3) if node.score else None
        }
        for node in result["source_nodes"]
    ]

    return {"answer": result["answer"], "sources": sources}


@app.get("/ask-physics")
async def ask_physics(query: str = Query(..., description="Ask any question from Class 10 Physics textbook")):
    try:
        result = get_cached_answer(query.strip().lower())
        return {
            "query": query,
            "answer": result["answer"],
            "sources": result["sources"]
        }
    except Exception as e:
        return {"error": str(e)}


# ========================= HYBRID QUESTION GENERATOR =========================
@lru_cache(maxsize=128)
def generate_hybrid_questions(topic: str):
    topic_normalized = topic.strip()

    # Retrieve real PYQs
    try:
        retrieved = pyq_retriever.retrieve(topic_normalized)
        real_pyqs = []
        for node in retrieved:
            if node.score < 0.3:
                continue
            text = node.text.strip()
            if len(text) < 30:
                continue
            real_pyqs.append({
                "question": text[:1500] + ("..." if len(text) > 1500 else ""),
                "year": node.metadata.get("primary_year") or ", ".join(node.metadata.get("years", ["Unknown"])),
                "source_file": node.metadata.get("source_file", "unknown.pdf"),
                "relevance_score": round(node.score, 3)
            })
        real_pyqs = sorted(real_pyqs, key=lambda x: x["relevance_score"], reverse=True)[:8]
    except Exception as e:
        print(f"PYQ retrieval failed: {e}")
        real_pyqs = []

    # Generate original questions
    prompt = f"""You are an expert CBSE Class 10 Physics teacher creating practice questions for board exam preparation.

Topic: {topic_normalized}

Generate 6 high-quality original questions:
- 2 Multiple Choice Questions (4 options, with correct answer and explanation)
- 1 Assertion-Reason type
- 1 Short Answer (2-3 marks)
- 1 Numerical problem (with solution steps)
- 1 Case-based / Diagram-based (describe diagram in text)

Return ONLY valid JSON:
{{
  "original_questions": [
    {{"type": "MCQ", "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "B", "explanation": "..."}},
    {{"type": "Assertion-Reason", "assertion": "...", "reason": "...", "answer": "Both A and R true, R explains A"}},
    {{"type": "Short Answer", "question": "...", "answer": "..."}},
    {{"type": "Numerical", "question": "...", "solution": "..."}},
    {{"type": "Case-based", "question": "...", "answer": "..."}}
  ]
}}"""

    try:
        response = Settings.llm.complete(prompt)
        raw = response.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        original_data = json.loads(raw)
        original_questions = original_data.get("original_questions", [])
    except Exception as e:
        print(f"Question generation failed: {e}")
        original_questions = [{"type": "Error", "question": "Failed to generate questions", "details": str(e)}]

    return {
        "real_pyqs_found": len(real_pyqs),
        "real_pyqs": real_pyqs,
        "original_questions": original_questions
    }


@app.get("/generate-questions")
async def generate_questions(
    topic: str = Query(..., description="Physics topic e.g., 'Ohm\\'s Law', 'Light Reflection'")
):
    try:
        result = generate_hybrid_questions(topic)
        return {
            "topic": topic,
            "real_pyqs_found": result["real_pyqs_found"],
            "real_pyqs": result["real_pyqs"],
            "original_questions": result["original_questions"],
            "tip": "Solve real PYQs first for exam pattern, then originals for deeper understanding!"
        }
    except Exception as e:
        return {"error": str(e)}


@app.get("/")
async def root():
    return {
        "message": "Class 10 Physics Assistant API is running!",
        "endpoints": ["/ask-physics", "/generate-questions", "/docs"]
    }


if __name__ == "__main__":
    import uvicorn
    print("Starting server...")
    print("→ http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
