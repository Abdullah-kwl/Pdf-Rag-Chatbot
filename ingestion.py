"""PDF ingestion: load, split, embed and persist chunks in Chroma."""

import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def get_embedding_model():
    """Return the sentence-transformer embedding model."""
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": config.EMBEDDING_DEVICE},
        encode_kwargs={"normalize_embeddings": config.NORMALISE_EMBEDDINGS},
    )


def get_vector_store(session_id, embedding_model):
    """Open or create the Chroma collection for a session."""
    persist_directory = config.DB_DIR / session_id
    persist_directory.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=str(persist_directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


def save_upload(uploaded_file, session_id):
    """Write an uploaded PDF to disk and return its path."""
    target_dir = config.UPLOAD_DIR / session_id
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / uploaded_file.name
    path.write_bytes(uploaded_file.getvalue())
    return path


def load_pdf(path):
    """Load a PDF into per-page documents with clean metadata."""
    documents = PyPDFLoader(str(path)).load()

    for doc in documents:
        doc.metadata["source"] = Path(path).name
        doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1

    documents = [doc for doc in documents if doc.page_content.strip()]

    if not documents:
        raise ValueError(f"No extractable text found in {Path(path).name}")

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def index_pdf(uploaded_file, session_id, vector_store):
    """Run the full ingestion pipeline for one uploaded PDF."""
    path = save_upload(uploaded_file, session_id)
    documents = load_pdf(path)
    chunks = split_documents(documents)
    vector_store.add_documents(chunks)
    return len(chunks)


def build_retriever(vector_store):
    """Return a retriever over the session collection."""
    return vector_store.as_retriever(
        search_type=config.RETRIEVER_SEARCH_TYPE,
        search_kwargs={
            "k": config.RETRIEVER_K,
            "score_threshold": config.SCORE_THRESHOLD,
        },
    )


def reset_session(session_id, vector_store=None):
    """Drop the session collection and delete its uploaded files."""
    if vector_store is not None:
        vector_store.delete_collection()

    shutil.rmtree(config.UPLOAD_DIR / session_id, ignore_errors=True)
