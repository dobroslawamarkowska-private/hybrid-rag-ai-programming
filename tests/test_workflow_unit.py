"""Testy jednostkowe workflow – bez wywołań LLM/API."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from workflow import (
    _parse_grader_response,
    _route_after_check,
    post_retrieval,
    RAGState,
    RELEVANCE_THRESHOLD,
)


class TestParseGraderResponse(unittest.TestCase):
    """Test parsowania odpowiedzi gradera (SCORE, REFINED)."""

    def test_valid_score_and_refined(self):
        text = "SCORE: 4\nREFINED: How to install Docker on Ubuntu?"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 4)
        self.assertEqual(refined, "How to install Docker on Ubuntu?")

    def test_score_below_threshold_with_refined(self):
        text = "SCORE: 2\nREFINED: Step-by-step Docker Engine installation for Linux"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 2)
        self.assertIn("Docker", refined)

    def test_score_1_min(self):
        text = "SCORE: 1\nREFINED: original question"
        score, _ = _parse_grader_response(text)
        self.assertEqual(score, 1)

    def test_score_5_max(self):
        text = "SCORE: 5\nREFINED: unchanged"
        score, _ = _parse_grader_response(text)
        self.assertEqual(score, 5)

    def test_score_out_of_range_clamped(self):
        score_0, _ = _parse_grader_response("SCORE: 0\nREFINED: x")
        score_99, _ = _parse_grader_response("SCORE: 99\nREFINED: x")
        self.assertEqual(score_0, 1)
        self.assertEqual(score_99, 5)

    def test_malformed_score_defaults_to_3(self):
        score, _ = _parse_grader_response("SCORE: abc\nREFINED: x")
        self.assertEqual(score, 3)

    def test_no_score_defaults_to_pass(self):
        score, refined = _parse_grader_response("Some random text\nREFINED: q")
        self.assertEqual(score, 3)
        self.assertEqual(refined, "q")

    def test_order_reversed_refined_first(self):
        text = "REFINED: improved query\nSCORE: 2"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 2)
        self.assertEqual(refined, "improved query")

    def test_empty_lines_ignored(self):
        text = "\nSCORE: 3\n\nREFINED:   trimmed  \n\n"
        score, refined = _parse_grader_response(text)
        self.assertEqual(score, 3)
        self.assertEqual(refined, "trimmed")


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

    def test_threshold_is_3(self):
        self.assertEqual(RELEVANCE_THRESHOLD, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
