"""Testy async retrieval i thread-safety (bez API – mock retrievera)."""

import asyncio
import concurrent.futures
import os
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from workflow import retrieval, retrieval_async, RAGState


def _make_doc(content: str, title: str = "Doc") -> Document:
    return Document(page_content=content, metadata={"title": title})


class TestRetrievalAsync(unittest.TestCase):
    """Test asynchronicznego retrieval."""

    def test_retrieval_async_returns_raw_docs(self):
        """retrieval_async zwraca raw_docs z poprawną strukturą."""
        fake_doc = _make_doc("Docker volume", "Volumes")
        mock_retriever = MagicMock()
        async def mock_ainvoke(_):
            return [fake_doc]
        mock_retriever.ainvoke = mock_ainvoke

        state: RAGState = {"expanded_queries": ["q1", "q2", "q3"]}
        with patch("workflow.get_shared_retriever", return_value=mock_retriever):
            result = asyncio.run(retrieval_async(state))

        self.assertIn("raw_docs", result)
        self.assertIsInstance(result["raw_docs"], list)
        self.assertGreaterEqual(len(result["raw_docs"]), 1)

    def test_retrieval_async_runs_tasks_in_parallel(self):
        """retrieval_async uruchamia ainvoke równolegle – różne query → różne wyniki."""
        def make_mock(query_to_doc):
            async def mock_ainvoke(query):
                return [query_to_doc[query]]
            return mock_ainvoke

        doc_a = _make_doc("Content A", "A")
        doc_b = _make_doc("Content B", "B")
        doc_c = _make_doc("Content C", "C")
        query_map = {"q1": doc_a, "q2": doc_b, "q3": doc_c}

        mock_retriever = MagicMock()
        mock_retriever.ainvoke = lambda q: asyncio.coroutine(lambda: [query_map[q]])()
        # Poprawny async mock – zwraca coroutine
        async def mock_ainvoke(query):
            return [query_map[query]]
        mock_retriever.ainvoke = mock_ainvoke

        state: RAGState = {"expanded_queries": ["q1", "q2", "q3"]}
        with patch("workflow.get_shared_retriever", return_value=mock_retriever):
            result = asyncio.run(retrieval_async(state))

        # Wszystkie 3 docs powinny być (różna treść = brak deduplikacji)
        titles = {d.metadata.get("title") for d in result["raw_docs"]}
        self.assertEqual(titles, {"A", "B", "C"})

    def test_retrieval_async_deduplicates_same_content(self):
        """retrieval_async deduplikuje dokumenty o tej samej treści."""
        same_doc = _make_doc("Same content", "Same")
        mock_retriever = MagicMock()
        async def mock_ainvoke(_):
            return [same_doc]
        mock_retriever.ainvoke = mock_ainvoke

        state: RAGState = {"expanded_queries": ["q1", "q2"]}
        with patch("workflow.get_shared_retriever", return_value=mock_retriever):
            result = asyncio.run(retrieval_async(state))

        self.assertEqual(len(result["raw_docs"]), 1)

    def test_retrieval_async_single_query(self):
        """retrieval_async z jednym query."""
        doc = _make_doc("Single", "S")
        mock_retriever = MagicMock()
        async def mock_ainvoke(_):
            return [doc]
        mock_retriever.ainvoke = mock_ainvoke

        state: RAGState = {"expanded_queries": ["only one"]}
        with patch("workflow.get_shared_retriever", return_value=mock_retriever):
            result = asyncio.run(retrieval_async(state))

        self.assertEqual(len(result["raw_docs"]), 1)
        self.assertEqual(result["raw_docs"][0].metadata["title"], "S")


class TestRetrievalThreadSafety(unittest.TestCase):
    """Test thread-safety – wiele wątków wywołuje retrieval równolegle."""

    def test_retrieval_concurrent_threads_no_error(self):
        """Wiele wątków wywołuje retrieval() równolegle – brak błędów, poprawne wyniki."""
        doc = _make_doc("Thread-safe doc", "TS")
        mock_retriever = MagicMock()
        async def mock_ainvoke(_):
            return [doc]
        mock_retriever.ainvoke = mock_ainvoke

        def run_retrieval():
            state: RAGState = {"expanded_queries": ["q1", "q2"]}
            with patch("workflow.get_shared_retriever", return_value=mock_retriever):
                return retrieval(state)

        n_threads = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(run_retrieval) for _ in range(n_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        for r in results:
            self.assertIn("raw_docs", r)
            self.assertIsInstance(r["raw_docs"], list)
            self.assertGreaterEqual(len(r["raw_docs"]), 1)

    def test_retrieval_concurrent_threads_isolated_results(self):
        """Każdy wątek dostaje poprawne, izolowane wyniki (mock per-thread)."""
        def make_mock():
            my_doc = _make_doc(f"Doc from thread {threading.current_thread().name}", "T")
            async def mock_ainvoke(_):
                return [my_doc]
            mock = MagicMock()
            mock.ainvoke = mock_ainvoke
            return mock

        results = []
        lock = threading.Lock()

        def run_retrieval():
            mock_retriever = make_mock()
            state: RAGState = {"expanded_queries": ["q"]}
            with patch("workflow.get_shared_retriever", return_value=mock_retriever):
                out = retrieval(state)
            with lock:
                results.append(out)
            return out

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: run_retrieval(), range(4)))

        self.assertEqual(len(results), 4)
        for r in results:
            self.assertEqual(len(r["raw_docs"]), 1)


class TestSharedRetrieverSingleton(unittest.TestCase):
    """Test get_shared_retriever – singleton, thread-safe init (mock Chroma/Embeddings)."""

    def test_get_shared_retriever_same_instance_from_multiple_threads(self):
        """Wiele wątków wywołuje get_shared_retriever – wszystkie dostają ten sam obiekt."""
        import retriever as retriever_mod
        from retriever import get_shared_retriever

        mock_instance = MagicMock()
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_instance

        with patch("retriever.Chroma", return_value=mock_vs):
            with patch("retriever.OpenAIEmbeddings"):
                with retriever_mod._SHARED_RETRIEVER_LOCK:
                    retriever_mod._SHARED_RETRIEVER = None

                ids = []
                lock = threading.Lock()

                def get_and_store():
                    r = get_shared_retriever(k=6)
                    with lock:
                        ids.append(id(r))
                    return r

                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    list(executor.map(lambda _: get_and_store(), range(10)))

                self.assertEqual(len(set(ids)), 1, "Wszystkie wątki powinny dostać ten sam obiekt")


if __name__ == "__main__":
    unittest.main(verbosity=2)
