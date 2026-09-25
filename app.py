"""
Simple RAG over Career PDFs — Streamlit app.

Pipeline:
PDFs -> cleaning -> chunking -> all-MiniLM-L6-v2 embeddings -> FAISS
-> retrieval -> cross-encoder reranking (RRF fusion) -> Groq LLM -> answer + sources.

Deployment: Streamlit Community Cloud (free tier).
LLM: Groq free-tier API (OpenAI-compatible).
API key is read from st.secrets, never hardcoded.
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
# Fixed pipeline configuration
# --------------------------------------------------------------------------

PDF_DIR = "data"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

TOP_K = 8
TOP_N_RERANK = 4

RRF_K = 60

# Current Groq model
GROQ_MODEL = "openai/gpt-oss-120b"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful assistant that answers ONLY using the provided context. "
    "Do not invent information and do not use outside knowledge. "
    "If the context does not contain enough information to answer, say exactly: "
    '"I could not find this information in the knowledge base." '
    "Be concise and clear. When possible, mention the source filename and page number."
)


# --------------------------------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Career Docs RAG",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------------------------------
# Text cleaning
# --------------------------------------------------------------------------

def clean_text(text: str) -> str:

    # Replace bullets and unusual characters
    text = re.sub(r"[\u2022\u25a0\u007f]", "-", text)

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove extra spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove spaces around line breaks
    text = re.sub(r" *\n *", "\n", text)

    return text.strip()


# --------------------------------------------------------------------------
# Cached model loading
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding + reranker models...")
def load_models():

    embedder = SentenceTransformer(EMBED_MODEL_NAME)

    reranker = CrossEncoder(RERANK_MODEL_NAME)

    return embedder, reranker


# --------------------------------------------------------------------------
# Build FAISS index
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner="Reading PDFs and building the FAISS index...")
def build_index(pdf_dir: str):

    embedder, _ = load_models()

    # Find all PDFs
    pdf_paths = sorted(
        glob.glob(
            os.path.join(pdf_dir, "*.pdf")
        )
    )

    if not pdf_paths:
        return [], None, []

    # ----------------------------------------------------------------------
    # 1. Extract text from PDFs
    # ----------------------------------------------------------------------

    raw_pages = []

    for path in pdf_paths:

        filename = os.path.basename(path)

        reader = PdfReader(path)

        for i, page in enumerate(reader.pages):

            text = page.extract_text() or ""

            raw_pages.append(
                {
                    "source": filename,
                    "page": i + 1,
                    "text": text
                }
            )

    # ----------------------------------------------------------------------
    # 2. Clean text
    # ----------------------------------------------------------------------

    cleaned_pages = []

    for p in raw_pages:

        ct = clean_text(p["text"])

        if ct:

            cleaned_pages.append(
                {
                    "source": p["source"],
                    "page": p["page"],
                    "text": ct
                }
            )

    # ----------------------------------------------------------------------
    # 3. Chunk text
    # ----------------------------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = []

    chunk_id = 0

    for p in cleaned_pages:

        pieces = splitter.split_text(p["text"])

        for piece in pieces:

            chunks.append(
                {
                    "text": piece.strip(),
                    "source": p["source"],
                    "page": p["page"],
                    "chunk_id": chunk_id,
                }
            )

            chunk_id += 1

    if not chunks:

        return [], None, [
            os.path.basename(p)
            for p in pdf_paths
        ]

    # ----------------------------------------------------------------------
    # 4. Create embeddings + FAISS index
    # ----------------------------------------------------------------------

    texts = [
        c["text"]
        for c in chunks
    ]

    embeddings = embedder.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=False
    ).astype("float32")

    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    # Create FAISS index
    index = faiss.IndexFlatIP(
        embeddings.shape[1]
    )

    index.add(embeddings)

    return (
        chunks,
        index,
        [
            os.path.basename(p)
            for p in pdf_paths
        ]
    )


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------

def retrieve(
    query,
    embedder,
    index,
    chunks,
    k=TOP_K
):

    # Convert query to embedding
    q_vec = embedder.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    # Normalize query
    faiss.normalize_L2(q_vec)

    # Search FAISS
    scores, indices = index.search(
        q_vec,
        k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:
            continue

        results.append(
            {
                **chunks[idx],
                "score": float(score)
            }
        )

    return results


# --------------------------------------------------------------------------
# Reranking with Reciprocal Rank Fusion
# --------------------------------------------------------------------------

def rerank(
    query,
    candidates,
    reranker,
    top_n=TOP_N_RERANK,
    rrf_k=RRF_K
):

    if not candidates:
        return []

    # Create query-document pairs
    pairs = [
        [query, c["text"]]
        for c in candidates
    ]

    # Cross-encoder scores
    rerank_scores = reranker.predict(
        pairs
    )

    for c, s in zip(
        candidates,
        rerank_scores
    ):

        c["rerank_score"] = float(s)

    # ----------------------------------------------------------------------
    # Ranking based on FAISS retrieval score
    # ----------------------------------------------------------------------

    by_retrieval = sorted(
        candidates,
        key=lambda c: c["score"],
        reverse=True
    )

    retrieval_rank = {
        c["chunk_id"]: i
        for i, c in enumerate(by_retrieval)
    }

    # ----------------------------------------------------------------------
    # Ranking based on cross-encoder score
    # ----------------------------------------------------------------------

    by_rerank = sorted(
        candidates,
        key=lambda c: c["rerank_score"],
        reverse=True
    )

    rerank_rank = {
        c["chunk_id"]: i
        for i, c in enumerate(by_rerank)
    }

    # ----------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ----------------------------------------------------------------------

    for c in candidates:

        r1 = retrieval_rank[
            c["chunk_id"]
        ]

        r2 = rerank_rank[
            c["chunk_id"]
        ]

        c["combined_score"] = (
            1.0 / (rrf_k + r1 + 1)
            +
            1.0 / (rrf_k + r2 + 1)
        )

    # Return top results
    return sorted(
        candidates,
        key=lambda c: c["combined_score"],
        reverse=True
    )[:top_n]


# --------------------------------------------------------------------------
# Build context for the LLM
# --------------------------------------------------------------------------

def build_context(reranked_chunks):

    parts = []

    for c in reranked_chunks:

        parts.append(
            f"[Source: {c['source']}, page {c['page']}]\n"
            f"{c['text']}"
        )

    return "\n\n---\n\n".join(parts)


# --------------------------------------------------------------------------
# Generate answer using Groq
# --------------------------------------------------------------------------

def generate_answer(
    query,
    context,
    api_key,
    model=GROQ_MODEL,
    temperature=0.0,
    timeout=30
):

    # Check API key
    if not api_key:

        return (
            "ERROR: GROQ_API_KEY is not configured. "
            "Add it to Streamlit Secrets."
        )

    # Build user prompt
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above."
    )

    # API payload
    payload = {

        "model": model,

        "messages": [

            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },

            {
                "role": "user",
                "content": user_prompt
            }

        ],

        "temperature": temperature,
    }

    # HTTP headers
    headers = {

        "Authorization": f"Bearer {api_key}",

        "Content-Type": "application/json"
    }

    try:

        # Call Groq API
        resp = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=timeout
        )

        # Raise error if request failed
        resp.raise_for_status()

        # Extract generated answer
        return resp.json()[
            "choices"
        ][0][
            "message"
        ][
            "content"
        ]

    except requests.exceptions.RequestException as e:

        # If Groq returned a detailed JSON error
        if getattr(e, "response", None) is not None:

            try:

                details = e.response.json()

                return (
                    f"ERROR calling Groq API: "
                    f"{details}"
                )

            except ValueError:

                return (
                    f"ERROR calling Groq API: "
                    f"{e.response.text}"
                )

        # General request error
        return (
            f"ERROR calling Groq API: {e}"
        )


# --------------------------------------------------------------------------
# Complete RAG pipeline
# --------------------------------------------------------------------------

def rag_pipeline(
    query,
    embedder,
    reranker,
    index,
    chunks,
    api_key,
    k=TOP_K,
    top_n=TOP_N_RERANK
):

    # 1. Retrieve relevant chunks
    candidates = retrieve(
        query,
        embedder,
        index,
        chunks,
        k=k
    )

    # 2. Rerank retrieved chunks
    reranked = rerank(
        query,
        candidates,
        reranker,
        top_n=top_n
    )

    # 3. Build context
    context = build_context(
        reranked
    )

    # 4. Generate final answer
    answer = generate_answer(
        query,
        context,
        api_key
    )

    # 5. Collect sources
    sources = sorted(
        {
            (
                c["source"],
                c["page"]
            )
            for c in reranked
        }
    )

    return (
        answer,
        sources,
        reranked
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.title(
    "📄 Career Docs RAG Assistant"
)

st.caption(
    "Ask questions about resumes, interviews, salary negotiation, "
    "and career growth. Answers are grounded only in the uploaded PDFs, "
    "with sources cited."
)


# --------------------------------------------------------------------------
# Get Groq API key from Streamlit Secrets
# --------------------------------------------------------------------------

api_key = st.secrets.get(
    "GROQ_API_KEY",
    os.environ.get(
        "GROQ_API_KEY",
        ""
    )
)


# --------------------------------------------------------------------------
# Load models and knowledge base
# --------------------------------------------------------------------------

embedder, reranker = load_models()

chunks, index, pdf_names = build_index(
    PDF_DIR
)


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:

    st.header(
        "Knowledge base"
    )

    if pdf_names:

        st.success(
            f"{len(pdf_names)} PDF(s) loaded, "
            f"{len(chunks)} chunks indexed."
        )

        for name in pdf_names:

            st.write(
                f"- {name}"
            )

    else:

        st.error(
            f"No PDFs found in `{PDF_DIR}/`. "
            "Add your PDF files there and redeploy."
        )

    if not api_key:

        st.warning(
            "GROQ_API_KEY is not set. "
            "Add it in Streamlit Secrets to enable answers."
        )

    st.divider()

    st.caption(
        f"Chunk size {CHUNK_SIZE} · "
        f"overlap {CHUNK_OVERLAP} · "
        f"Top-K {TOP_K} · "
        f"Top-N {TOP_N_RERANK}"
    )

    st.caption(
        f"LLM: Groq · {GROQ_MODEL}"
    )


# --------------------------------------------------------------------------
# Chat history
# --------------------------------------------------------------------------

if "history" not in st.session_state:

    st.session_state.history = []


# --------------------------------------------------------------------------
# Display previous conversations
# --------------------------------------------------------------------------

for turn in st.session_state.history:

    with st.chat_message("user"):

        st.write(
            turn["query"]
        )

    with st.chat_message("assistant"):

        st.write(
            turn["answer"]
        )

        if turn["sources"]:

            with st.expander(
                "Sources"
            ):

                for s, p in turn["sources"]:

                    st.write(
                        f"- {s}, page {p}"
                    )


# --------------------------------------------------------------------------
# Chat input
# --------------------------------------------------------------------------

query = st.chat_input(
    "Ask a question about the career documents..."
)


# --------------------------------------------------------------------------
# Process new question
# --------------------------------------------------------------------------

if query:

    # Display user question
    with st.chat_message("user"):

        st.write(
            query
        )

    # Generate assistant response
    with st.chat_message("assistant"):

        if index is None:

            st.error(
                "No knowledge base is loaded — "
                "add PDFs to the `data/` folder."
            )

        elif not api_key:

            st.error(
                "GROQ_API_KEY is not configured "
                "in Streamlit Secrets."
            )

        else:

            with st.spinner(
                "Retrieving, reranking, and generating..."
            ):

                answer, sources, reranked = rag_pipeline(
                    query,
                    embedder,
                    reranker,
                    index,
                    chunks,
                    api_key
                )

            # Display answer
            st.write(
                answer
            )

            # Display sources
            if sources:

                with st.expander(
                    "Sources"
                ):

                    for s, p in sources:

                        st.write(
                            f"- {s}, page {p}"
                        )

            # Save conversation
            st.session_state.history.append(
                {
                    "query": query,
                    "answer": answer,
                    "sources": sources
                }
            )
