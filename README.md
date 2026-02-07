# Hybrid RAG – AI Programming

Advanced RAG (Retrieval Augmented Generation) pipeline dla dokumentacji Docker. Zbudowany w **LangGraph** z etapami: pre-retrieval, retrieval, grader + refinement, post-retrieval, generate.

## Wymagania

- Python 3.12+
- Klucz API OpenRouter (LLM i embeddingi przez openrouter.ai)
- Opcjonalnie: konto Kaggle (do pobrania datasetu przy budowaniu indeksu)

## Instalacja

```bash
# Środowisko wirtualne
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Zależności
pip install -r req.txt

# Konfiguracja – utwórz plik .env
# OPENROUTER_API_KEY=sk-or-v1-...
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
# Dla LangSmith: LANGSMITH_TRACING=true, LANGSMITH_API_KEY, LANGSMITH_PROJECT
```

## Uruchomienie

```bash
# Budowanie indeksu (jednorazowo; wymaga danych – parquet w ./data lub Kaggle)
python build_index.py

# Zapytanie
python -c "from workflow import ask; print(ask('Jak zainstalować Docker?'))"

# Uruchomienie z przykładowym pytaniem
python workflow.py

# Testy (pełne – wymaga indeksu i API)
python -m unittest discover tests -v

# Tylko testy jednostkowe (bez API)
SKIP_INTEGRATION=1 python -m unittest discover tests -v
```

## Architektura

Szczegółowy opis przepływu, diagramy i konfiguracja – zobacz [docs/ADVANCED_RAG.md](docs/ADVANCED_RAG.md).

## Struktura projektu

| Plik / katalog | Opis |
|----------------|------|
| `config.py` | Konfiguracja: ścieżki Chroma, modele |
| `build_index.py` | Budowanie indeksu wektorowego z dokumentacji Docker |
| `retriever.py` | Retriever i tool do wyszukiwania w dokumentacji |
| `workflow.py` | LangGraph workflow RAG |
| `tests/` | Testy retrievera i workflow |
| `docs/ADVANCED_RAG.md` | Pełna dokumentacja architektury |
