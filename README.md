# Career Docs RAG Assistant

A simple, explainable Retrieval-Augmented Generation (RAG) app over 7 career-advice PDFs
(resume writing, job descriptions, interviews, salary negotiation, career growth, data roles).

**Pipeline:**

```
PDFs → cleaning → chunking → all-MiniLM-L6-v2 embeddings → FAISS
     → retrieval (Top-K) → cross-encoder reranking (Top-N, via RRF) → Groq LLM → answer + sources
```

## Files

| File | Purpose |
|---|---|
| `Simple_RAG_7_PDFs_Final.ipynb` | Cleaned, end-to-end notebook (Colab or local Jupyter) — build & test the pipeline step by step |
| `app.py` | Streamlit app implementing the same pipeline for deployment |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keeps secrets and local artifacts out of git |
| `data/` (you create this) | Put your PDF files here — read by `app.py` |

## What changed from the original notebook

1. **Ollama removed entirely.** No local model server, no `apt-get`/`curl` install cells — that
   only works in a live Colab VM, not on Streamlit Community Cloud.
2. **LLM replaced with the Groq free-tier API** (`llama-3.3-70b-versatile`), an OpenAI-compatible
   REST endpoint. No credit card needed, and it's fast enough for interactive use. The API key is
   read from **Streamlit Secrets** (`app.py`) or an environment variable typed in at runtime via
   `getpass` (notebook) — it is **never hardcoded**.
3. **Reranker bug fixed.** For "What are important ATS resume formatting rules?", FAISS correctly
   retrieved the right chunk (`chunk_id=2`, retrieval score ≈ 0.544), but the cross-encoder scored
   it a very negative ≈ -10.5 — a scale outlier that knocked it out of the Top-N under the old
   min-max score blending. Fixed with **Reciprocal Rank Fusion (RRF)**: chunks are combined by
   *rank* in each list, not by raw score, so a single outlier score can no longer dominate the
   result. This is a standard, well-known IR technique (used e.g. by Elasticsearch hybrid search)
   — simple to explain, no extra models or heuristics needed.
4. **Notebook decluttered.** Removed duplicate debug/inspection cells, redundant `evaluate_retrieval`
   re-runs, and stray config re-definitions (e.g. `CHUNK_SIZE` was defined twice with different
   values in the original). The notebook now runs cleanly top-to-bottom with **Runtime → Run all**.
5. **Pipeline hyperparameters kept exactly as requested:** `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`,
   `TOP_K=8`, `TOP_N_RERANK=4`. Embeddings (`all-MiniLM-L6-v2`), FAISS (`IndexFlatIP` on normalized
   vectors), and the cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) are unchanged.
6. Answers stay **grounded only in retrieved context** (same system prompt as before), and every
   answer is shown with its **PDF filename + page number** sources.

## Run the notebook (Colab)

1. Upload `Simple_RAG_7_PDFs_Final.ipynb` to Google Colab.
2. Upload your 7 PDFs into the same Colab file browser folder.
3. Runtime → Run all. When prompted, paste a free Groq API key from
   https://console.groq.com/keys (no credit card required).

## Run the app locally

```bash
pip install -r requirements.txt
mkdir data
# copy your 7 PDFs into data/
mkdir -p .streamlit
echo 'GROQ_API_KEY = "your-groq-key-here"' > .streamlit/secrets.toml
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (free)

1. **Get a free Groq API key:** https://console.groq.com/keys (no credit card needed).
2. **Create a GitHub repo** containing: `app.py`, `requirements.txt`, `.gitignore`, and a `data/`
   folder with your 7 PDFs (PDFs are small text documents, fine to commit — or use Git LFS if
   you prefer).
3. Go to https://share.streamlit.io → **New app** → pick your repo, branch, and set the main file
   path to `app.py`.
4. Before (or right after) the first deploy, open **App settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your-groq-key-here"
   ```
5. Click **Deploy**. First boot downloads the embedding + reranker models and builds the FAISS
   index (cached via `st.cache_resource`, so it only happens once per app restart).
6. Open the app URL, ask a question, and check that answers show sources (filename + page).

## Test questions

The notebook's `evaluation_questions` list (also exercised manually in the app) includes:

- What are the five steps of the system design framework?
- What is the STAR method?
- What are the three tiers of requirements in a job description?
- What should I consider when negotiating salary?
- What are the main data roles and their roadmaps?
- **What are important ATS resume formatting rules?** (regression test for the reranker fix)
- What are the fundamentals of career growth and personal branding?
- How should I use a job description's requirements to rewrite my resume bullets? (cross-document)

## Notes

- Swap `GROQ_MODEL` in `app.py` / the notebook config cell for `llama-3.1-8b-instant` if you want
  faster/cheaper responses at a small quality cost — both are on Groq's free tier.
- If you outgrow Groq's free-tier rate limits (30 requests/min, ~14,400/day at time of writing),
  Cerebras and OpenRouter offer comparable free, OpenAI-compatible tiers with minimal code changes
  (just swap the base URL / model name in `generate_answer`).
