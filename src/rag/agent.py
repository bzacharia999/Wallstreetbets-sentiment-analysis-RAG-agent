"""
RAG agent — Ollama-powered retrieval-augmented generation for WSB Q&A.
"""

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma

from src.utils.config import OLLAMA_MODEL, OLLAMA_BASE_URL


# ── Prompt ──────────────────────────────────────────────────────────────────

WSB_SYSTEM_PROMPT = """You are a knowledgeable analyst of the r/wallstreetbets subreddit.
You answer questions based ONLY on the provided context from real WSB posts.
If the context doesn't contain enough information to answer, say so honestly.
Reference specific posts when possible. You understand WSB terminology
(diamond hands, tendies, YOLO, apes, moon, bagholder, etc.).

Context:
{context}

Question: {question}

Answer:"""

_PROMPT = PromptTemplate(
    template=WSB_SYSTEM_PROMPT,
    input_variables=["context", "question"],
)


# ── Chain construction ──────────────────────────────────────────────────────

def create_rag_chain(
    vector_store: Chroma,
    model: str | None = None,
    temperature: float = 0.3,
) -> RetrievalQA:
    """
    Build a RetrievalQA chain using a local Ollama model.

    Parameters
    ----------
    vector_store : Chroma
        The populated ChromaDB vector store.
    model : str, optional
        Ollama model name (default from config, typically ``llama3.2``).
    temperature : float
        LLM sampling temperature.
    """
    model = model or OLLAMA_MODEL

    retriever = vector_store.as_retriever(
        search_type="mmr",  # Maximal Marginal Relevance for diverse results
        search_kwargs={"k": 5},
    )

    llm = ChatOllama(
        model=model,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": _PROMPT},
    )

    return chain


# ── Query helper ────────────────────────────────────────────────────────────

def ask(chain: RetrievalQA, question: str) -> dict:
    """
    Query the RAG chain.

    Returns
    -------
    dict
        answer : str
        sources : list[dict] — each with content, author, score, sentiment.
    """
    response = chain.invoke({"query": question})

    sources = []
    for doc in response.get("source_documents", []):
        sources.append(
            {
                "content": doc.page_content[:300],
                "author": doc.metadata.get("author", "unknown"),
                "score": doc.metadata.get("score", 0),
                "sentiment": doc.metadata.get("sentiment", ""),
                "topic": doc.metadata.get("topic", ""),
            }
        )

    return {
        "answer": response["result"],
        "sources": sources,
    }
