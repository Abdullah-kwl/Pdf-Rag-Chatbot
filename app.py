"""Streamlit chatbot that answers questions over PDFs uploaded in the sidebar."""

import os
import uuid

import streamlit as st
from dotenv import load_dotenv

import config
import ingestion
from graph import build_graph

load_dotenv(config.ENV_PATH)

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads weights)...")
def cached_embedding_model():
    """Load the embedding model once per Streamlit process."""
    return ingestion.get_embedding_model()


def init_state():
    """Seed per-session state."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex[:12]
        st.session_state.messages = []
        st.session_state.indexed = {}
        st.session_state.vector_store = None
        st.session_state.graph = None


def ensure_pipeline():
    """Create the vector store and graph on first use."""
    if st.session_state.vector_store is None:
        embedding_model = cached_embedding_model()
        st.session_state.vector_store = ingestion.get_vector_store(
            st.session_state.session_id, embedding_model
        )
        retriever = ingestion.build_retriever(st.session_state.vector_store)
        st.session_state.graph = build_graph(retriever)


def index_uploads(uploaded_files):
    """Index any uploaded PDF that is not already in the collection."""
    pending = [f for f in uploaded_files if f.name not in st.session_state.indexed]
    if not pending:
        return

    ensure_pipeline()

    for uploaded_file in pending:
        with st.spinner(f"Indexing {uploaded_file.name}..."):
            try:
                chunk_count = ingestion.index_pdf(
                    uploaded_file,
                    st.session_state.session_id,
                    st.session_state.vector_store,
                )
                st.session_state.indexed[uploaded_file.name] = chunk_count
            except Exception as exc:
                st.error(f"{uploaded_file.name}: {exc}")


def clear_knowledge_base():
    """Drop the collection, uploaded files and chat history."""
    ingestion.reset_session(st.session_state.session_id, st.session_state.vector_store)
    st.session_state.indexed = {}
    st.session_state.vector_store = None
    st.session_state.graph = None
    st.session_state.messages = []


def render_sidebar():
    """Draw the upload panel and controls."""
    with st.sidebar:
        st.header("Documents")

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )

        if uploaded_files:
            index_uploads(uploaded_files)

        if st.session_state.indexed:
            st.caption("Indexed")
            for name, chunk_count in st.session_state.indexed.items():
                st.write(f"- {name} — {chunk_count} chunks")
        else:
            st.caption("No documents indexed yet.")

        st.divider()

        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.session_id = uuid.uuid4().hex[:12]
            st.rerun()

        if st.button("Clear knowledge base", width="stretch"):
            clear_knowledge_base()
            st.rerun()

        st.divider()
        st.caption(
            f"Embeddings: {config.EMBEDDING_MODEL}  \n"
            f"LLM: {config.LLM_MODEL}  \n"
            f"Chunks: {config.CHUNK_SIZE}/{config.CHUNK_OVERLAP}  \n"
            f"Top-k: {config.RETRIEVER_K} @ score ≥ {config.SCORE_THRESHOLD}"
        )


def stream_answer(question):
    """Stream generation tokens from the graph for one turn."""
    graph_config = {"configurable": {"thread_id": st.session_state.session_id}}

    for chunk, metadata in st.session_state.graph.stream(
        {"messages": [{"role": "user", "content": question}]},
        graph_config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") == "generate" and chunk.content:
            yield chunk.content


def get_sources():
    """Return the documents retrieved for the latest turn."""
    graph_config = {"configurable": {"thread_id": st.session_state.session_id}}
    return st.session_state.graph.get_state(graph_config).values.get("documents", [])


def render_sources(documents):
    """Show retrieved passages in a collapsed panel."""
    if not documents:
        return

    with st.expander(f"Sources ({len(documents)})"):
        for i, doc in enumerate(documents, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**[{i}] {source} — p.{page}**")
            st.caption(doc.page_content)


def main():
    init_state()
    render_sidebar()

    st.title("PDF RAG Chatbot")

    if not os.getenv("GROQ_API_KEY"):
        st.warning(f"GROQ_API_KEY not found. Add it to {config.ENV_PATH}.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("sources", []))

    ready = bool(st.session_state.indexed)
    placeholder = "Ask about your documents" if ready else "Upload a PDF to begin"

    question = st.chat_input(placeholder, disabled=not ready)
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(question))
            documents = get_sources()
            render_sources(documents)
        except Exception as exc:
            st.error(f"Generation failed: {exc}")
            return

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": documents}
    )


if __name__ == "__main__":
    main()
