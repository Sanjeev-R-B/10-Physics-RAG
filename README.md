<<<<<<< HEAD
# ⚡ PhysicsIQ — Class 10 AI Physics Tutor

An AI-powered study assistant for Class 10 Physics, built on a RAG (Retrieval-Augmented Generation) pipeline using NCERT textbook content. Students can ask questions and get detailed, textbook-grounded answers, or generate exam-style practice questions on any topic.

**Live Demo → [your-app.streamlit.app](https://your-app.streamlit.app)**

---

## Features

- **Ask Physics** — Ask any Class 10 Physics question and get a detailed answer with formula, explanation, and real-world example, sourced directly from NCERT chapters
- **Practice Questions** — Generate topic-wise practice sets combining real Previous Year Questions (PYQs) with AI-generated MCQs, Numericals, Assertion-Reason, and Case-based questions
- **Source transparency** — Every answer shows which textbook section and PDF it came from, with a relevance score

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Vector Database | Pinecone |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| RAG Framework | LlamaIndex |
| PDF Parsing | LlamaParse |

---

## Project Structure

```
science_llm/
├── streamlit_app.py          # Main Streamlit UI
├── requirements.txt
├── scripts/
│   ├── parse_physics_pdf.py  # Parse PDFs → parsed.jsonl
│   ├── chunk_and_enrich.py   # Chunk text → chunks.jsonl
│   ├── build_index.py        # Upload chunks → Pinecone
│   ├── chunk_pyqs.py         # Process PYQ PDFs
│   └── index_pyqs.py         # Upload PYQs → Pinecone
└── app_fastapi.py            # (Optional) REST API version
```

---

## How It Works

```
PDF Textbooks
     │
     ▼
parse_physics_pdf.py  →  parsed.jsonl
     │
     ▼
chunk_and_enrich.py   →  chunks.jsonl  (512-token chunks with metadata)
     │
     ▼
build_index.py        →  Pinecone Vector DB
     │
     ▼
streamlit_app.py      →  User Query
                              │
                    Embed query (BGE)
                              │
                    Pinecone similarity search
                              │
                    Filter by relevance score (≥0.62)
                              │
                    Groq LLM synthesizes answer
```

---

## Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/science_llm.git
cd science_llm
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up `.env`
```bash
cp .env.example .env
# Fill in your keys
```

```env
PINECONE_API_KEY=your_pinecone_key
GROQ_API_KEY=your_groq_key
PINECONE_INDEX_NAME=physics-textbook
PINECONE_ENVIRONMENT=us-east-1
```

### 5. Build the index (first time only)
```bash
python scripts/parse_physics_pdf.py
python scripts/chunk_and_enrich.py
python scripts/build_index.py
```

### 6. Run the app
```bash
streamlit run streamlit_app.py
```

---

## Deploying to Streamlit Cloud

1. Push this repo to GitHub (see `.gitignore` — PDFs and keys are excluded)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo and set main file as `streamlit_app.py`
4. Go to **Settings → Secrets** and add:
```toml
PINECONE_API_KEY = "your_key"
GROQ_API_KEY = "your_key"
```
5. Click **Deploy**

---

## API Keys (all free)

| Service | Free Tier | Link |
|---|---|---|
| Groq | 14,400 requests/day | [console.groq.com](https://console.groq.com) |
| Pinecone | 2GB storage, 1 index | [pinecone.io](https://pinecone.io) |

---

## Future Improvements

- Add more subjects (Chemistry, Biology, Maths)
- Chapter-wise filtering in the UI
- Voice input for questions
- Performance analytics dashboard for students

---

## Author

**Sanjeev**
Built as part of an AI/ML project — Class 10 Physics RAG system using modern LLM tooling.
=======

