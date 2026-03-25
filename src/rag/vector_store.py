"""
Vector store operations — document preparation, chunking, and ChromaDB persistence.
"""

import shutil
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from src.utils.config import CHROMA_DIR, EMBEDDING_MODEL


def prepare_documents(df: pd.DataFrame) -> list[Document]:
    """
    Convert a DataFrame of scraped/analyzed posts into LangChain Documents.

    Each document's page_content includes title + selftext.
    Metadata carries post_id, author, score, flair, sentiment, topic, etc.
    """
    docs: list[Document] = []
    for _, row in df.iterrows():
        content = f"Title: {row['title']}\n\n{row.get('selftext', '')}"
        metadata = {
            "post_id": str(row.get("id", "")),
            "author": str(row.get("author", "unknown")),
            "score": int(row.get("score", 0)),
            "flair": str(row.get("flair", "")),
            "created_utc": str(row.get("created_utc", "")),
            "sentiment": str(row.get("sentiment_label", "")),
            "topic": str(row.get("topic_label", "")),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def chunk_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return the project-standard embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(chunks: list[Document]) -> Chroma:
    """
    Embed document chunks and persist to a local ChromaDB store.

    If a previous store exists it is deleted first to avoid duplicates.
    """
    # Clear old store if it exists
    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=_get_embeddings(),
        persist_directory=CHROMA_DIR,
    )
    return vector_store


def load_vector_store() -> Chroma:
    """Load an existing persisted ChromaDB store."""
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=_get_embeddings(),
    )


def build_index(df: pd.DataFrame) -> Chroma:
    """
    End-to-end helper: prepare → chunk → embed → persist.
    Returns the ChromaDB vector store.
    """
    docs = prepare_documents(df)
    chunks = chunk_documents(docs)
    return create_vector_store(chunks)
