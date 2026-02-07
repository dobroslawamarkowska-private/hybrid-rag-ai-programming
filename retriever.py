"""Retriever tool for searching Docker docs chunks. Loads existing index (no indexing)."""

from dotenv import load_dotenv

load_dotenv(override=True)  # przed importem LangChain (LangSmith observability)

from langchain_chroma import Chroma
from langchain_core.tools import create_retriever_tool
from langchain_openai import OpenAIEmbeddings

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL


def get_retriever(k: int = 4):
    """Ładuje istniejący indeks Chroma i zwraca retriever. Nie buduje indeksu."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    vs = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
    return vs.as_retriever(search_kwargs={"k": k})


def create_docker_docs_tool():
    """Zwraca LangChain tool do wyszukiwania chunków dokumentacji Docker."""
    retriever = get_retriever()
    return create_retriever_tool(
        retriever,
        "search_docker_docs",
        "Wyszukuj fragmenty dokumentacji Docker (instrukcje, API, konfiguracja, opisy).",
    )
