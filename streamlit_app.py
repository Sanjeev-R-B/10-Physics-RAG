import os
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PhysicsIQ — Class 10 AI Tutor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
# After st.set_page_config(), add these 4 lines:
import streamlit as st

# Works locally (.env) AND on Streamlit Cloud (secrets.toml)
if "PINECONE_API_KEY" in st.secrets:
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark science-lab theme */
.stApp {
    background: #0a0e1a;
    color: #e2e8f0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0f1629 !important;
    border-right: 1px solid #1e2d4a;
}

/* Hero header */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.hero-sub {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 2rem;
    letter-spacing: 0.02em;
}

/* Answer card */
.answer-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-left: 4px solid #60a5fa;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 1rem 0;
    line-height: 1.8;
    color: #e2e8f0;
    font-size: 1rem;
}

/* Source badge */
.source-badge {
    display: inline-block;
    background: #1e2d4a;
    border: 1px solid #2d4a7a;
    color: #60a5fa;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    margin: 4px 4px 4px 0;
    font-family: 'Space Grotesk', sans-serif;
}

.score-high { border-color: #34d399; color: #34d399; background: #0d2818; }
.score-mid  { border-color: #fbbf24; color: #fbbf24; background: #2a1f08; }
.score-low  { border-color: #f87171; color: #f87171; background: #2a0f0f; }

/* Section label */
.section-label {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.6rem;
}

/* Question chip */
.q-chip {
    background: #1e2d4a;
    border: 1px solid #2d4a7a;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    cursor: pointer;
    font-size: 0.9rem;
    color: #93c5fd;
    transition: all 0.2s;
}

/* MCQ card */
.mcq-card {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    margin: 0.8rem 0;
}

.mcq-correct {
    color: #34d399;
    font-weight: 600;
}

.tag {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 0.6rem;
}
.tag-mcq      { background: #1e3a5f; color: #60a5fa; }
.tag-ar       { background: #2a1f5a; color: #a78bfa; }
.tag-short    { background: #0d2818; color: #34d399; }
.tag-num      { background: #2a1a08; color: #fbbf24; }
.tag-case     { background: #2a0f1a; color: #f472b6; }

/* Input styling */
.stTextInput > div > div > input {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #60a5fa !important;
    box-shadow: 0 0 0 2px rgba(96,165,250,0.2) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* Divider */
hr { border-color: #1e2d4a !important; }

/* Tabs */
button[data-baseweb="tab"] {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    color: #475569 !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #60a5fa !important;
}

/* Metric cards */
.metric-box {
    background: #111827;
    border: 1px solid #1e3a5f;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-lbl {
    font-size: 0.78rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# INIT: Embeddings + Pinecone + Groq
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading knowledge base...")
def init_resources():
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.embed_model = embed_model

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    tb_store = PineconeVectorStore(pinecone_index=pc.Index("physics-textbook"))
    tb_index = VectorStoreIndex.from_vector_store(vector_store=tb_store)
    tb_retriever = tb_index.as_retriever(similarity_top_k=6)

    pyq_store = PineconeVectorStore(pinecone_index=pc.Index("physics-pyqs"))
    pyq_index = VectorStoreIndex.from_vector_store(vector_store=pyq_store)
    pyq_retriever = pyq_index.as_retriever(similarity_top_k=10)

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    return tb_retriever, pyq_retriever, groq_client


# ─────────────────────────────────────────────
# CORE: Ask Physics with Groq
# ─────────────────────────────────────────────
SCORE_THRESHOLD = 0.62

def ask_physics(query: str, tb_retriever, groq_client):
    nodes = tb_retriever.retrieve(query)

    relevant = [n for n in nodes if n.score and n.score >= SCORE_THRESHOLD]
    if not relevant:
        relevant = nodes[:2]

    context = "\n\n---\n\n".join([n.node.get_content() for n in relevant])

    prompt = f"""You are an expert Class 10 Physics teacher. Use ONLY the context below to answer.

Instructions:
- Give a clear, detailed explanation
- Include formula with each symbol explained (if applicable)  
- Explain step by step
- Give a real-world example
- Do NOT add "This topic is not covered" if you have already answered

Context:
{context}

Question: {query}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content.strip()
    sources = [
        {
            "section": n.node.metadata.get("section", "Unknown"),
            "source_file": n.node.metadata.get("source", "unknown.pdf"),
            "score": round(n.score, 3) if n.score else 0,
        }
        for n in relevant
    ]
    return answer, sources


# ─────────────────────────────────────────────
# CORE: Generate Questions with Groq
# ─────────────────────────────────────────────
def generate_questions(topic: str, pyq_retriever, groq_client):
    import json

    # Real PYQs
    pyq_nodes = pyq_retriever.retrieve(topic)
    real_pyqs = []
    for n in pyq_nodes:
        if n.score and n.score < 0.3:
            continue
        text = n.text.strip()
        if len(text) < 30:
            continue
        real_pyqs.append({
            "question": text[:1000],
            "year": n.metadata.get("primary_year", "Unknown"),
            "score": round(n.score, 3) if n.score else 0,
        })
    real_pyqs = sorted(real_pyqs, key=lambda x: x["score"], reverse=True)[:6]

    # Original questions
    prompt = f"""You are a CBSE Class 10 Physics teacher. Generate 5 exam-level questions on: {topic}

Include:
- 2 MCQs (4 options, correct answer, explanation)
- 1 Short Answer (2-3 marks)
- 1 Numerical (with solution)
- 1 Assertion-Reason

Return ONLY valid JSON (no markdown, no extra text):
{{"questions": [
  {{"type": "MCQ", "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "A", "explanation": "..."}},
  {{"type": "MCQ", "question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct": "B", "explanation": "..."}},
  {{"type": "Short Answer", "question": "...", "answer": "..."}},
  {{"type": "Numerical", "question": "...", "solution": "..."}},
  {{"type": "Assertion-Reason", "assertion": "...", "reason": "...", "answer": "..."}}
]}}"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",

        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    try:
        data = json.loads(raw)
        original_qs = data.get("questions", [])
    except Exception:
        original_qs = []

    return real_pyqs, original_qs


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 0.5rem 0;'>
        <span style='font-family: Space Grotesk; font-size: 1.4rem; font-weight: 700;
                     background: linear-gradient(135deg,#60a5fa,#a78bfa);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            ⚡ PhysicsIQ
        </span>
        <div style='color:#475569; font-size:0.8rem; margin-top:2px;'>Class 10 AI Tutor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-label">Navigate</div>', unsafe_allow_html=True)

    page = st.radio(
        "",
        ["💬 Ask Physics", "📝 Practice Questions"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Chapters</div>', unsafe_allow_html=True)
    chapters = [
        "⚡ Electricity",
        "🧲 Magnetic Effects",
        "💡 Light – Reflection & Refraction",
        "👁️ Human Eye",
        "🌿 Our Environment",
    ]
    for ch in chapters:
        st.markdown(f"<div style='color:#475569; font-size:0.85rem; padding:4px 0;'>{ch}</div>",
                    unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='color:#334155; font-size:0.75rem; line-height:1.6;'>
        Powered by<br>
        <span style='color:#60a5fa;'>Groq</span> · 
        <span style='color:#a78bfa;'>Pinecone</span> · 
        <span style='color:#34d399;'>LlamaIndex</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────
try:
    tb_retriever, pyq_retriever, groq_client = init_resources()
except Exception as e:
    st.error(f"Failed to connect to services: {e}")
    st.stop()


# ─────────────────────────────────────────────
# PAGE: ASK PHYSICS
# ─────────────────────────────────────────────
if "💬" in page:
    st.markdown('<div class="hero-title">Ask your Physics question</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Answers grounded in your Class 10 NCERT textbook</div>',
                unsafe_allow_html=True)

    # Sample questions
    st.markdown('<div class="section-label">Try these</div>', unsafe_allow_html=True)
    samples = [
        "What is Ohm's Law? Give the formula.",
        "How does a concave mirror form images?",
        "What is electromagnetic induction?",
        "Explain the structure of the human eye.",
    ]
    cols = st.columns(2)
    for i, q in enumerate(samples):
        if cols[i % 2].button(q, key=f"sample_{i}"):
            st.session_state["prefill"] = q

    st.markdown("---")

    prefill = st.session_state.pop("prefill", "")
    query = st.text_input(
        "Your question",
        value=prefill,
        placeholder="e.g. What is Ohm's Law?",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 5])
    ask_clicked = col1.button("Ask ⚡", use_container_width=True)

    if ask_clicked and query.strip():
        with st.spinner("Searching textbook..."):
            try:
                answer, sources = ask_physics(query, tb_retriever, groq_client)

                # Answer
                st.markdown('<div class="section-label" style="margin-top:1.5rem;">Answer</div>',
                            unsafe_allow_html=True)
                st.markdown(f'<div class="answer-card">{answer}</div>', unsafe_allow_html=True)

                # Sources
                st.markdown('<div class="section-label" style="margin-top:1.2rem;">Sources used</div>',
                            unsafe_allow_html=True)
                for s in sources:
                    score = s["score"]
                    cls = "score-high" if score >= 0.7 else "score-mid" if score >= 0.6 else "score-low"
                    st.markdown(
                        f'<span class="source-badge {cls}">'
                        f'{s["section"]} &nbsp;·&nbsp; {s["source_file"]} &nbsp;·&nbsp; {score}'
                        f'</span>',
                        unsafe_allow_html=True
                    )

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    elif ask_clicked:
        st.warning("Please enter a question first.")

    # Chat history in session
    if "history" not in st.session_state:
        st.session_state.history = []

    if ask_clicked and query.strip():
        try:
            st.session_state.history.insert(0, {"q": query, "a": answer})
            st.session_state.history = st.session_state.history[:5]
        except Exception:
            pass

    if st.session_state.get("history"):
        st.markdown("---")
        st.markdown('<div class="section-label">Recent questions</div>', unsafe_allow_html=True)
        for item in st.session_state.history[1:]:
            with st.expander(item["q"]):
                st.markdown(f'<div class="answer-card">{item["a"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: PRACTICE QUESTIONS
# ─────────────────────────────────────────────
elif "📝" in page:
    st.markdown('<div class="hero-title">Practice Questions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real PYQs + AI-generated exam questions by topic</div>',
                unsafe_allow_html=True)

    topic = st.text_input(
        "Topic",
        placeholder="e.g. Ohm's Law, Light Reflection, Magnetic Effects",
        label_visibility="collapsed"
    )

    col1, col2 = st.columns([1, 5])
    gen_clicked = col1.button("Generate 📝", use_container_width=True)

    if gen_clicked and topic.strip():
        with st.spinner("Generating questions..."):
            try:
                real_pyqs, original_qs = generate_questions(topic, pyq_retriever, groq_client)

                tab1, tab2 = st.tabs([
                    f"🗂️ Previous Year Questions ({len(real_pyqs)})",
                    f"🤖 AI-Generated ({len(original_qs)})"
                ])

                with tab1:
                    if real_pyqs:
                        for i, q in enumerate(real_pyqs, 1):
                            with st.expander(f"Q{i}. {q['question'][:80]}..."):
                                st.markdown(q["question"])
                                st.markdown(
                                    f'<span class="source-badge">Year: {q["year"]}</span>'
                                    f'<span class="source-badge">Score: {q["score"]}</span>',
                                    unsafe_allow_html=True
                                )
                    else:
                        st.info("No PYQs found for this topic. Try a broader term.")

                with tab2:
                    TYPE_TAGS = {
                        "MCQ": "tag-mcq",
                        "Short Answer": "tag-short",
                        "Numerical": "tag-num",
                        "Assertion-Reason": "tag-ar",
                        "Case-based": "tag-case",
                    }
                    for i, q in enumerate(original_qs, 1):
                        qtype = q.get("type", "Question")
                        tag_cls = TYPE_TAGS.get(qtype, "tag-mcq")

                        with st.expander(f"Q{i}. [{qtype}] {q.get('question', q.get('assertion', ''))[:70]}..."):
                            st.markdown(
                                f'<span class="tag {tag_cls}">{qtype}</span>',
                                unsafe_allow_html=True
                            )

                            if qtype == "MCQ":
                                st.markdown(f"**{q.get('question', '')}**")
                                for opt in q.get("options", []):
                                    correct = q.get("correct", "")
                                    if opt.startswith(correct):
                                        st.markdown(f'<span class="mcq-correct">✓ {opt}</span>',
                                                    unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"&nbsp;&nbsp;{opt}")
                                if q.get("explanation"):
                                    st.info(f"💡 {q['explanation']}")

                            elif qtype == "Assertion-Reason":
                                st.markdown(f"**Assertion:** {q.get('assertion', '')}")
                                st.markdown(f"**Reason:** {q.get('reason', '')}")
                                st.success(f"**Answer:** {q.get('answer', '')}")

                            elif qtype == "Numerical":
                                st.markdown(f"**{q.get('question', '')}**")
                                st.markdown(f"**Solution:** {q.get('solution', '')}")

                            else:
                                st.markdown(f"**{q.get('question', '')}**")
                                if q.get("answer"):
                                    st.markdown(f"**Answer:** {q['answer']}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    elif gen_clicked:
        st.warning("Please enter a topic first.")
