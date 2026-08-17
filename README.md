# PDF RAG Chatbot

Streamlit chatbot that answers questions over PDFs uploaded in the sidebar. Retrieval over a
Chroma vector store with bge-m3 embeddings, generation via Groq, orchestrated as a LangGraph graph.

## Structure

| File | Purpose |
| --- | --- |
| `config.py` | All tunables as module constants |
| `ingestion.py` | Load PDF → chunk → embed → Chroma; retriever construction |
| `graph.py` | LangGraph pipeline: `condense → retrieve → generate` |
| `app.py` | Streamlit UI, upload panel, streaming chat |

## Graph

```
START → condense → retrieve → generate → END
```

- `condense` rewrites a follow-up into a standalone query using the last
  `HISTORY_WINDOW` messages. Skipped on the first turn.
- `retrieve` runs a cosine similarity search with a relevance-score floor, so
  off-topic questions return nothing rather than noise.
- `generate` answers strictly from the context block and cites `[file, p.N]`.

Conversation state lives in a `MemorySaver` checkpointer keyed by `thread_id`
(the Streamlit session id); `st.session_state.messages` is the render copy.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `GROQ_API_KEY`.

## Run

```powershell
streamlit run app.py
```

First launch downloads the bge-m3 weights (~2.3 GB) unless already cached. It runs on
CPU by default — switch `EMBEDDING_DEVICE` to `cuda` in `config.py` for GPU embedding.

## Notes

- Each session gets its own Chroma collection under `db/<session_id>/`, so uploads
  from separate sessions never mix. "Clear knowledge base" drops the collection and
  deletes the stored uploads.
- Re-uploading a file with the same name in one session is skipped, not re-indexed.
- Scanned PDFs with no text layer raise an error at index time; they need OCR first.
- `SCORE_THRESHOLD = 0.3` assumes normalised embeddings and cosine distance. Lower it
  if valid questions come back as "not covered in the uploaded documents".
