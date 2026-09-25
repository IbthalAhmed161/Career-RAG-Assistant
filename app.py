"""
Simple RAG over Career PDFs — Streamlit app.

Pipeline: PDFs -> cleaning -> chunking -> all-MiniLM-L6-v2 embeddings -> FAISS
-> retrieval -> cross-encoder reranking (RRF fusion) -> Groq LLM -> answer + sources.

Deployment: Streamlit Community Cloud (free tier).
LLM: Groq free-tier API (OpenAI-compatible). API key is read from st.secrets,
never hardcoded.
"""

import os
import re
import glob

import numpy as np
import requests
import streamlit as st
import faiss
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder

# --------------------------------------------------------------------------
# Fixed pipeline configuration (kept identical to the notebook)
# --------------------------------------------------------------------------
PDF_DIR = "data"  # put your PDFs in a "data/" folder next to this file
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 8
TOP_N_RERANK = 4
RRF_K = 60  # Reciprocal Rank Fusion constant (see notebook Section 9 for why)

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = (
    "You are a careful assistant that answers ONLY using the provided context. "
    "Do not invent information and do not use outside knowledge. "
    "If the context does not contain enough information to answer, say exactly: "
    '"I could not find this information in the knowledge base." '
    "Be concise and clear. When possible, mention the source filename and page number."
)

st.set_page_config(page_title="Career Docs RAG", page_icon="📄", layout="wide")


# --------------------------------------------------------------------------
# Text cleaning (identical to notebook)
# --------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = re.sub(r"[\u2022\u25a0\u007f]", "-", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


# --------------------------------------------------------------------------
# Cached model loading (loaded once per server, shared across users)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding + reranker models...")
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    reranker = CrossEncoder(RERANK_MODEL_NAME)
    return embedder, reranker


# --------------------------------------------------------------------------
# Cached index build (runs once per server, until the PDFs change)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Reading PDFs and building the FAISS index...")
def build_index(pdf_dir: str):
    embedder, _ = load_models()

    pdf_paths = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_paths:
        return [], None, []

    # 1. Extract
    raw_pages = []
    for path in pdf_paths:
        filename = os.path.basename(path)
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            raw_pages.append({"source": filename, "page": i + 1, "text": text})

    # 2. Clean
    cleaned_pages = []
    for p in raw_pages:
        ct = clean_text(p["text"])
        if ct:
            cleaned_pages.append({"source": p["source"], "page": p["page"], "text": ct})

    # 3. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    chunk_id = 0
    for p in cleaned_pages:
        for piece in splitter.split_text(p["text"]):
            chunks.append({
                "text": piece.strip(),
                "source": p["source"],
                "page": p["page"],
                "chunk_id": chunk_id,
            })
            chunk_id += 1

    if not chunks:
        return [], None, [os.path.basename(p) for p in pdf_paths]

    # 4. Embed + FAISS
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return chunks, index, [os.path.basename(p) for p in pdf_paths]


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
def retrieve(query, embedder, index, chunks, k=TOP_K):
    q_vec = embedder.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_vec)
    scores, indices = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        results.append({**chunks[idx], "score": float(score)})
    return results


# --------------------------------------------------------------------------
# Reranking with Reciprocal Rank Fusion (robust to cross-encoder outlier scores)
# --------------------------------------------------------------------------
def rerank(query, candidates, reranker, top_n=TOP_N_RERANK, rrf_k=RRF_K):
    if not candidates:
        return []

    pairs = [[query, c["text"]] for c in candidates]
    rerank_scores = reranker.predict(pairs)
    for c, s in zip(candidates, rerank_scores):
        c["rerank_score"] = float(s)

    by_retrieval = sorted(candidates, key=lambda c: c["score"], reverse=True)
    retrieval_rank = {c["chunk_id"]: i for i, c in enumerate(by_retrieval)}

    by_rerank = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    rerank_rank = {c["chunk_id"]: i for i, c in enumerate(by_rerank)}

    for c in candidates:
        r1 = retrieval_rank[c["chunk_id"]]
        r2 = rerank_rank[c["chunk_id"]]
        c["combined_score"] = 1.0 / (rrf_k + r1 + 1) + 1.0 / (rrf_k + r2 + 1)

    return sorted(candidates, key=lambda c: c["combined_score"], reverse=True)[:top_n]


def build_context(reranked_chunks):
    parts = []
    for c in reranked_chunks:
        parts.append(f"[Source: {c['source']}, page {c['page']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------
# Generation via Groq (free-tier, OpenAI-compatible API)
# --------------------------------------------------------------------------
def generate_answer(query, context, api_key, model=GROQ_MODEL, temperature=0.0, timeout=30):
    if not api_key:
        return "ERROR: GROQ_API_KEY is not configured. Add it to Streamlit Secrets."

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer using only the context above."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        return f"ERROR calling Groq API: {e}"


def rag_pipeline(query, embedder, reranker, index, chunks, api_key, k=TOP_K, top_n=TOP_N_RERANK):
    candidates = retrieve(query, embedder, index, chunks, k=k)
    reranked = rerank(query, candidates, reranker, top_n=top_n)
    context = build_context(reranked)
    answer = generate_answer(query, context, api_key)
    sources = sorted({(c["source"], c["page"]) for c in reranked})
    return answer, sources, reranked


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("📄 Career Docs RAG Assistant")
st.caption(
    "Ask questions about resumes, interviews, salary negotiation, and career growth. "
    "Answers are grounded only in the uploaded PDFs, with sources cited."
)

api_key = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

embedder, reranker = load_models()
chunks, index, pdf_names = build_index(PDF_DIR)

with st.sidebar:
    st.header("Knowledge base")
    if pdf_names:
        st.success(f"{len(pdf_names)} PDF(s) loaded, {len(chunks)} chunks indexed.")
        for name in pdf_names:
            st.write(f"- {name}")
    else:
        st.error(f"No PDFs found in `{PDF_DIR}/`. Add your PDF files there and redeploy.")

    if not api_key:
        st.warning("GROQ_API_KEY is not set. Add it in Streamlit Secrets to enable answers.")

    st.divider()
    st.caption(f"Chunk size {CHUNK_SIZE} · overlap {CHUNK_OVERLAP} · Top-K {TOP_K} · Top-N {TOP_N_RERANK}")
    st.caption(f"LLM: Groq · {GROQ_MODEL}")

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn["sources"]:
            with st.expander("Sources"):
                for s, p in turn["sources"]:
                    st.write(f"- {s}, page {p}")

query = st.chat_input("Ask a question about the career documents...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        if index is None:
            st.error("No knowledge base is loaded — add PDFs to the `data/` folder.")
        elif not api_key:
            st.error("GROQ_API_KEY is not configured in Streamlit Secrets.")
        else:
            with st.spinner("Retrieving, reranking, and generating..."):
                answer, sources, reranked = rag_pipeline(
                    query, embedder, reranker, index, chunks, api_key
                )
            st.write(answer)
            if sources:
                with st.expander("Sources"):
                    for s, p in sources:
                        st.write(f"- {s}, page {p}")
            st.session_state.history.append({"query": query, "answer": answer, "sources": sources})
