"""Build Docker docs index on demand. Run: python build_index.py"""

import os
import json
import shutil

from dotenv import load_dotenv

load_dotenv(override=True)

import pandas as pd
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL

PARQUET_FILENAME = "docker_docs_rag.parquet"
PARQUET_PATH = os.path.join(os.path.dirname(__file__), PARQUET_FILENAME)
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _download_from_kaggle():
    """Pobiera dataset z Kaggle (kagglehub)."""
    dataset = os.environ.get("KAGGLE_DATASET", "martininf1n1ty/docker-docs-rag-dataset")
    import kagglehub

    path = kagglehub.dataset_download(dataset)
    print("Path to dataset files:", path)
    local = os.path.join(path, PARQUET_FILENAME)
    if os.path.isfile(local):
        return local
    for root, _, files in os.walk(path):
        for f in files:
            if f == PARQUET_FILENAME or f.endswith(".parquet"):
                return os.path.join(root, f)
    return None


def _load_dataframe():
    """Wczytuje DataFrame z parquet (lokalnie lub Kaggle)."""
    df = pd.DataFrame()
    if os.path.isfile(PARQUET_PATH):
        df = pd.read_parquet(PARQUET_PATH)
        print("✅ Dataset wczytany z pliku parquet")
    if df.empty:
        local_path = os.path.join(DATA_DIR, PARQUET_FILENAME)
        if os.path.isfile(local_path):
            df = pd.read_parquet(local_path)
            print("✅ Dataset wczytany z pliku parquet (data/)")
        else:
            print("📥 Próba pobrania z Kaggle...")
            downloaded = _download_from_kaggle()
            if downloaded:
                df = pd.read_parquet(downloaded)
                print("✅ Dataset pobrany z Kaggle")
            else:
                print("❌ Nie znaleziono pliku parquet.")
    return df


def _chroma_safe_metadata_value(value):
    """Convert value to a type Chroma accepts: str, int, float, bool, or None."""
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value
    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
        return json.dumps(value, ensure_ascii=False) if value else ""
    except (TypeError, ValueError):
        return str(value)


def _df_to_docs(df: pd.DataFrame):
    """Zamienia DataFrame na listę Document."""
    docs = []
    for _, row in df.iterrows():
        content = row.get("content", "")
        if not content or (isinstance(content, str) and not content.strip()):
            title = row.get("title", "")
            desc = row.get("description", "")
            content = f"{title}\n\n{desc}".strip() or "(brak treści)"
        if isinstance(content, str) and not content.strip():
            continue
        metadata = {
            "file_path": _chroma_safe_metadata_value(row.get("file_path", "")),
            "title": _chroma_safe_metadata_value(row.get("title", "")),
        }
        for key in ("tags", "keywords", "aliases"):
            if key in row and row[key] is not None:
                metadata[key] = _chroma_safe_metadata_value(row[key])
        docs.append(Document(page_content=content, metadata=metadata))
    return docs


def build_index():
    """Buduje indeks Chroma z dokumentacji Docker. Zapisuje do CHROMA_DIR."""
    force_rebuild = os.environ.get("REBUILD_INDEX", "").lower() in ("1", "true", "yes")
    if force_rebuild and os.path.isdir(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print("🔄 REBUILD_INDEX=1 — usunięto stary indeks, budowanie od zera...")

    df = _load_dataframe()
    docs = _df_to_docs(df)
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=400,
        chunk_overlap=100,
    )
    doc_splits = text_splitter.split_documents(docs) if docs else []

    if not doc_splits:
        doc_splits = [Document(page_content="(brak dokumentów)", metadata={})]

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
    )
    os.makedirs(os.path.dirname(CHROMA_DIR) or ".", exist_ok=True)
    Chroma.from_documents(
        doc_splits,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )
    print(f"✅ Indeks Chroma zbudowany: {len(doc_splits):,} chunków → {CHROMA_DIR}")


if __name__ == "__main__":
    _download_from_kaggle()
    build_index()
