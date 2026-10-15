---
name: RAG Agent
description: How to build a Retrieval-Augmented Generation agent that answers user queries using r/wallstreetbets context
---

# RAG Agent Skill

## Overview
Build a **Retrieval-Augmented Generation (RAG)** pipeline that:
1. Chunks and embeds scraped WSB posts into a vector store
2. Retrieves relevant posts given a user query
3. Feeds retrieved context to an LLM to generate grounded answers

---

## Prerequisites

```bash
pip install langchain langchain-openai langchain-community chromadb sentence-transformers python-dotenv
```

For local/offline LLM support (optional):
```bash
pip install langchain-ollama
# + install Ollama from https://ollama.com
```

---

## Implementation Steps

### 1. Document Preparation & Chunking

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd

def prepare_documents(df: pd.DataFrame) -> list[Document]:
    """Convert DataFrame rows into LangChain Documents with metadata."""
    docs = []
    for _, row in df.iterrows():
        content = f"Title: {row['title']}\n\n{row.get('selftext', '')}"
        metadata = {
            "post_id": row["id"],
            "author": row.get("author", "unknown"),
            "score": row.get("score", 0),
            "flair": row.get("flair", ""),
            "created_utc": str(row.get("created_utc", "")),
            "sentiment": row.get("sentiment_label", ""),
            "topic": row.get("topic_label", ""),
        }
        docs.append(Document(page_content=content, metadata=metadata))
    return docs

def chunk_documents(docs: list[Document]) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
```

### 2. Vector Store (ChromaDB)

```python
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = "data/chroma_db"

def create_vector_store(chunks: list[Document]) -> Chroma:
    """Embed chunks and persist to local ChromaDB."""
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
    )
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    return vector_store

def load_vector_store() -> Chroma:
    """Load an existing ChromaDB store."""
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )
```

### 3. RAG Chain

```python
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

WSB_SYSTEM_PROMPT = """You are a knowledgeable analyst of the r/wallstreetbets subreddit. 
You answer questions based ONLY on the provided context from real WSB posts.
If the context doesn't contain enough information to answer, say so honestly.
Reference specific posts when possible. Understand WSB terminology (diamond hands, 
tendies, YOLO, apes, etc.).

Context:
{context}

Question: {question}

Answer:"""

def create_rag_chain(vector_store: Chroma, model_name: str = "gpt-4o-mini"):
    """Create a RetrievalQA chain."""
    retriever = vector_store.as_retriever(
        search_type="mmr",          # Maximal Marginal Relevance for diversity
        search_kwargs={"k": 5},
    )

    llm = ChatOpenAI(
        model=model_name,
        temperature=0.3,
    )

    prompt = PromptTemplate(
        template=WSB_SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    return chain
```

### 4. Ollama (Local LLM Alternative)

```python
from langchain_ollama import ChatOllama

def create_local_rag_chain(vector_store: Chroma, model: str = "llama3.2"):
    """Use a locally-running Ollama model instead of OpenAI."""
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    llm = ChatOllama(model=model, temperature=0.3)

    prompt = PromptTemplate(
        template=WSB_SYSTEM_PROMPT,
        input_variables=["context", "question"],
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
```

### 5. Query Interface

```python
def ask(chain, question: str) -> dict:
    """
    Query the RAG chain.
    Returns: {"result": str, "source_documents": list[Document]}
    """
    response = chain.invoke({"query": question})
    return {
        "answer": response["result"],
        "sources": [
            {
                "content": doc.page_content[:200],
                "author": doc.metadata.get("author"),
                "score": doc.metadata.get("score"),
                "sentiment": doc.metadata.get("sentiment"),
            }
            for doc in response.get("source_documents", [])
        ],
    }
```

---

## Key Considerations

| Topic | Guidance |
|-------|----------|
| **Chunk size** | 500 tokens with 50 overlap balances retrieval precision and context size. Adjust if answers lack context. |
| **Retrieval strategy** | `MMR` (Maximal Marginal Relevance) prevents retrieving 5 near-identical chunks. |
| **Metadata filtering** | ChromaDB supports `where` filters — e.g., retrieve only `sentiment=positive` posts. |
| **Cost** | `gpt-4o-mini` is ~$0.15/1M input tokens. For zero-cost, use Ollama with `llama3.2`. |
| **Embedding consistency** | Use the **same** embedding model for indexing and querying. |
| **Re-indexing** | When scraping new data, clear and rebuild the vector store rather than appending duplicates. |
| **Streamlit integration** | Cache the vector store and chain with `@st.cache_resource`. Use `st.chat_message` for the chat UI. |

---

## Verification

1. Index a small set of ≥10 posts and confirm ChromaDB persists to disk.
2. Query with a relevant question (e.g., "What are people saying about GME?") and confirm retrieved sources are relevant.
3. Confirm the LLM answer references actual post content and doesn't hallucinate.
4. Test with a question not covered by the data — confirm the model says it doesn't have enough information.
5. Test both OpenAI and Ollama backends (if configured).
