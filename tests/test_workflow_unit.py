"""Testy jednostkowe workflow – bez wywołań LLM/API."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from workflow import (
    _parse_grader_response,
    _route_after_check,
    check_and_refine_query,
    post_retrieval,
    retrieval,
    RAGState,
    RELEVANCE_THRESHOLD,
)


class TestParseGraderResponse(unittest.TestCase):
    """Test parsowania odpowiedzi gradera (SCORE 0.00–1.00, REFINED)."""

    def test_valid_score_and_refined(self):
        text = "SCORE: 0.75\nREFINED: How to install Docker on Ubuntu?"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 0.75)
        self.assertEqual(refined, "How to install Docker on Ubuntu?")

    def test_score_below_threshold_with_refined(self):
        text = "SCORE: 0.35\nREFINED: Step-by-step Docker Engine installation for Linux"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 0.35)
        self.assertIn("Docker", refined)

    def test_score_0_min(self):
        text = "SCORE: 0.00\nREFINED: original question"
        score, _ = _parse_grader_response(text)
        self.assertEqual(score, 0.0)

    def test_score_1_max(self):
        text = "SCORE: 1.00\nREFINED: unchanged"
        score, _ = _parse_grader_response(text)
        self.assertEqual(score, 1.0)

    def test_score_out_of_range_clamped(self):
        score_neg, _ = _parse_grader_response("SCORE: -0.5\nREFINED: x")
        score_over, _ = _parse_grader_response("SCORE: 1.5\nREFINED: x")
        self.assertEqual(score_neg, 0.0)
        self.assertEqual(score_over, 1.0)

    def test_malformed_score_defaults_to_pass(self):
        score, _ = _parse_grader_response("SCORE: abc\nREFINED: x")
        self.assertEqual(score, 0.5)

    def test_no_score_defaults_to_pass(self):
        score, refined = _parse_grader_response("Some random text\nREFINED: q")
        self.assertEqual(score, 0.5)
        self.assertEqual(refined, "q")

    def test_order_reversed_refined_first(self):
        text = "REFINED: improved query\nSCORE: 0.42"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 0.42)
        self.assertEqual(refined, "improved query")

    def test_empty_lines_ignored(self):
        text = "\nSCORE: 0.60\n\nREFINED:   trimmed  \n\n"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 0.6)
        self.assertEqual(refined, "trimmed")


class TestCheckAndRefine(unittest.TestCase):
    """Test check_and_refine – zachowanie przy refinement."""

    def test_refine_preserves_original_query(self):
        """Przy refinemenci nie nadpisujemy query – generate odpowiada na oryginalne pytanie."""
        state: RAGState = {
            "query": "How to install Docker Hub on Linux?",
            "raw_docs": [Document(page_content="x" * 100, metadata={"title": "Doc"})],
            "retrieval_attempt": 0,
        }
        fake_response = MagicMock(content="SCORE: 0.35\nREFINED: How to sign in to Docker Hub?")
        fake_chain = MagicMock()
        fake_chain.invoke = MagicMock(return_value=fake_response)
        mock_prompt_instance = MagicMock()
        mock_prompt_instance.__or__ = MagicMock(return_value=fake_chain)
        with patch("workflow.ChatPromptTemplate") as MockPrompt:
            MockPrompt.from_messages.return_value = mock_prompt_instance
            out = check_and_refine_query(state)
        self.assertIn("expanded_queries", out)
        self.assertEqual(out["expanded_queries"], ["How to sign in to Docker Hub?"])
        self.assertEqual(out["retrieval_attempt"], 1)
        self.assertNotIn("query", out, "query must not be overwritten – generate needs original")

    def test_skip_no_docs_detail(self):
        """Skipped (no docs) gdy brak raw_docs."""
        state: RAGState = {"query": "q", "raw_docs": [], "retrieval_attempt": 0, "trace": True}
        out = check_and_refine_query(state)
        self.assertEqual(out["flow_log"][0]["detail"], "Skipped (no docs)")

    def test_skip_retry_limit_detail(self):
        """Skipped (retry limit reached) gdy attempt >= 1."""
        state: RAGState = {
            "query": "q",
            "raw_docs": [Document(page_content="x", metadata={"title": "T"})],
            "retrieval_attempt": 1,
            "trace": True,
        }
        out = check_and_refine_query(state)
        self.assertIn("retry limit reached", out["flow_log"][0]["detail"])


class TestRouteAfterCheck(unittest.TestCase):
    """Test routingu po check_and_refine (retry vs post_retrieval)."""

    def test_retrieval_attempt_1_goes_to_retrieval(self):
        state: RAGState = {"retrieval_attempt": 1}
        self.assertEqual(_route_after_check(state), "retrieval")

    def test_retrieval_attempt_0_goes_to_post_retrieval(self):
        state: RAGState = {"retrieval_attempt": 0}
        self.assertEqual(_route_after_check(state), "post_retrieval")

    def test_missing_attempt_goes_to_post_retrieval(self):
        state: RAGState = {}
        self.assertEqual(_route_after_check(state), "post_retrieval")


class TestPostRetrieval(unittest.TestCase):
    """Test post_retrieval – budowanie kontekstu z dokumentów (bez LLM)."""

    def _make_doc(self, content: str, title: str = "Test") -> Document:
        return Document(page_content=content, metadata={"title": title})

    def test_builds_context_from_docs(self):
        docs = [
            self._make_doc("Content A", "Doc1"),
            self._make_doc("Content B", "Doc2"),
        ]
        state: RAGState = {"raw_docs": docs}
        out = post_retrieval(state)
        self.assertIn("reranked_docs", out)
        self.assertIn("context", out)
        self.assertIn("Content A", out["context"])
        self.assertIn("Content B", out["context"])
        self.assertIn("Doc1", out["context"])
        self.assertIn("Doc2", out["context"])

    def test_limits_to_6_chunks_in_context(self):
        docs = [self._make_doc(f"Content {i}", f"Doc{i}") for i in range(10)]
        state: RAGState = {"raw_docs": docs}
        out = post_retrieval(state)
        self.assertEqual(len(out["reranked_docs"]), 8)  # reranked keeps 8
        # context uses first 6
        for i in range(6):
            self.assertIn(f"Content {i}", out["context"])
        self.assertNotIn("Content 6", out["context"])

    def test_empty_docs_still_returns_context(self):
        state: RAGState = {"raw_docs": []}
        out = post_retrieval(state)
        self.assertEqual(out["reranked_docs"], [])
        self.assertEqual(out["context"], "")


class TestRelevanceThreshold(unittest.TestCase):
    """Stałe workflow."""

    def test_threshold_is_0_5(self):
        self.assertEqual(RELEVANCE_THRESHOLD, 0.5)


class TestRetrievalWorkers(unittest.TestCase):
    """Test retrieval z równoległymi workerami – mock retrievera."""

    def test_retrieval_returns_raw_docs_with_workers(self):
        """Retrieval (async + shared retriever) zwraca raw_docs z deduplikacją."""
        fake_doc = Document(page_content="Docker volume persist", metadata={"title": "Volumes"})
        mock_retriever = MagicMock()
        mock_retriever.ainvoke = MagicMock(return_value=[fake_doc])
        # ainvoke returns awaitable – use async mock
        async def mock_ainvoke(_):
            return [fake_doc]
        mock_retriever.ainvoke = mock_ainvoke

        state: RAGState = {"expanded_queries": ["query1", "query2"]}
        with patch("workflow.get_shared_retriever", return_value=mock_retriever):
            out = retrieval(state)

        self.assertIn("raw_docs", out)
        self.assertIsInstance(out["raw_docs"], list)
        # 2 queries × 1 doc each, deduplicated if same content
        self.assertGreaterEqual(len(out["raw_docs"]), 1)
        self.assertLessEqual(len(out["raw_docs"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
