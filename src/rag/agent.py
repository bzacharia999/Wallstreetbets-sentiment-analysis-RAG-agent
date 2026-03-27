"""
RAG agent — Ollama-powered retrieval-augmented generation for WSB Q&A.
Uses LangChain Expression Language (LCEL) for the chain.
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma

from src.utils.config import OLLAMA_MODEL, OLLAMA_BASE_URL



WSB_SYSTEM_PROMPT = """You are a knowledgeable analyst of the r/wallstreetbets subreddit.
You answer questions based ONLY on the provided context from real WSB posts.
If the context doesn't contain enough information to answer, say so honestly.
Reference specific posts when possible. You understand WSB terminology
(diamond hands, tendies, YOLO, apes, moon, bagholder, etc.).

Context:
{context}

Question: {question}"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful WSB analyst. Answer based only on the provided context."),
    ("human", WSB_SYSTEM_PROMPT),
])


def _format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


# Wrapper to hold chain + retriever together

class RAGChain:
    """Wraps the LCEL chain and retriever so we can access source docs."""

    def __init__(self, chain, retriever):
        self.chain = chain
        self.retriever = retriever

    def invoke(self, question: str) -> str:
        return self.chain.invoke(question)

    def get_source_docs(self, question: str):
        return self.retriever.invoke(question)


# Chain construction

def create_rag_chain(
    vector_store: Chroma,
    model: str | None = None,
    temperature: float = 0.3,
) -> RAGChain:
    """
    Build an LCEL RAG chain using a local Ollama model.

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

    # LCEL chain: retrieve → format → prompt → LLM → parse
    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | llm
        | StrOutputParser()
    )

    return RAGChain(chain=chain, retriever=retriever)


# Query helper

def ask(rag: RAGChain, question: str) -> dict:
    """
    Query the RAG chain.

    Returns
    -------
    dict
        answer : str
        sources : list[dict] — each with content, author, score, sentiment.
    """
    # Get the answer
    answer = rag.invoke(question)

    # Retrieve source docs separately for display
    source_docs = rag.get_source_docs(question)

    sources = []
    for doc in source_docs:
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
        "answer": answer,
        "sources": sources,
    }
