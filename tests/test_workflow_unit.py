"""Testy jednostkowe workflow – bez wywołań LLM/API."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from workflow import (
    _parse_grader_response,
    _route_after_check,
    ingest,
    post_retrieval,
    retrieval,
    summarize_conversation,
    generate,
    RAGState,
    RELEVANCE_THRESHOLD,
    MAX_MESSAGES_BEFORE_SUMMARY,
    MAX_CONTENT_TOKENS,
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


class TestIngest(unittest.TestCase):
    """Test ingest – pobieranie query z messages."""

    def test_extracts_query_from_last_human_message(self):
        state: RAGState = {
            "messages": [
                HumanMessage(content="Jak zainstalować Docker?"),
            ]
        }
        out = ingest(state)
        self.assertEqual(out["query"], "Jak zainstalować Docker?")

    def test_multi_turn_uses_last_human_message(self):
        state: RAGState = {
            "messages": [
                HumanMessage(content="Q1"),
                AIMessage(content="A1"),
                HumanMessage(content="Q2 – ostatnie pytanie"),
            ]
        }
        out = ingest(state)
        self.assertEqual(out["query"], "Q2 – ostatnie pytanie")

    def test_empty_messages_uses_state_query(self):
        state: RAGState = {"query": "fallback query"}
        out = ingest(state)
        self.assertEqual(out["query"], "fallback query")

    def test_empty_messages_no_query_returns_empty_string(self):
        state: RAGState = {}
        out = ingest(state)
        self.assertEqual(out["query"], "")

    def test_last_message_aimessage_falls_back_to_state_query(self):
        state: RAGState = {
            "messages": [HumanMessage(content="Q"), AIMessage(content="A")],
            "query": "from state",
        }
        out = ingest(state)
        self.assertEqual(out["query"], "from state")


class TestSummarizeConversation(unittest.TestCase):
    """Test summarize_conversation – gdy messages > 2 i > MAX_MESSAGES_BEFORE_SUMMARY tokenów."""

    def test_returns_empty_when_messages_leq_2(self):
        state: RAGState = {
            "messages": [HumanMessage(content="Q1"), AIMessage(content="A1")],
        }
        out = summarize_conversation(state)
        self.assertEqual(out, {})

    def test_returns_empty_when_tokens_below_threshold(self):
        """3 messages, ale mało tokenów – bez summarization."""
        state: RAGState = {
            "messages": [
                HumanMessage(content="Q1"),
                AIMessage(content="A1"),
                HumanMessage(content="Q2"),
            ],
        }
        with patch("workflow.count_tokens_approximately", return_value=100):
            out = summarize_conversation(state)
        self.assertEqual(out, {})

    def test_summarizes_when_tokens_exceed_threshold(self):
        """Mock LLM – sprawdź że zwraca summary i RemoveMessage (gdy messages mają id)."""
        msg1 = HumanMessage(content="Question 1 about Docker", id="msg1")
        msg2 = AIMessage(content="Answer 1", id="msg2")
        msg3 = HumanMessage(content="Question 2", id="msg3")
        state: RAGState = {
            "messages": [msg1, msg2, msg3],
        }
        mock_llm = MagicMock()
        mock_llm.bind.return_value.invoke.return_value = MagicMock(content="Summary of Q1, A1, Q2")

        with patch("workflow.count_tokens_approximately", return_value=2000):
            with patch("workflow._get_smart_llm", return_value=mock_llm):
                out = summarize_conversation(state)

        self.assertIn("summary", out)
        self.assertEqual(out["summary"], "Summary of Q1, A1, Q2")
        self.assertIn("messages", out)
        # to_remove = messages[:-2] = [msg1]; RemoveMessage dla msg1
        self.assertEqual(len(out["messages"]), 1)
        mock_llm.bind.return_value.invoke.assert_called_once()


class TestGenerate(unittest.TestCase):
    """Test generate – z mockowanym LLM."""

    def test_includes_summary_in_system_when_present(self):
        state: RAGState = {
            "context": "Docker volume docs",
            "summary": "User asked about volumes",
            "messages": [HumanMessage(content="How to persist data?")],
        }
        mock_response = MagicMock()
        mock_response.content = "Use volumes."
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("workflow._get_smart_llm", return_value=mock_llm):
            out = generate(state)

        self.assertEqual(out["answer"], "Use volumes.")
        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        self.assertIn("Conversation summary", system_content)
        self.assertIn("User asked about volumes", system_content)

    def test_works_without_summary(self):
        state: RAGState = {
            "context": "Docker docs",
            "messages": [HumanMessage(content="Question")],
        }
        mock_response = MagicMock()
        mock_response.content = "Answer"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_response

        with patch("workflow._get_smart_llm", return_value=mock_llm):
            out = generate(state)

        self.assertEqual(out["answer"], "Answer")
        call_args = mock_llm.invoke.call_args[0][0]
        system_content = call_args[0].content
        self.assertNotIn("Conversation summary", system_content)


class TestWorkflowConstants(unittest.TestCase):
    """Stałe workflow."""

    def test_max_messages_before_summary(self):
        self.assertEqual(MAX_MESSAGES_BEFORE_SUMMARY, 1500)

    def test_max_content_tokens(self):
        self.assertEqual(MAX_CONTENT_TOKENS, 4096)


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
        """Retrieval (orchestrator–workers) zwraca raw_docs z deduplikacją."""
        fake_doc = Document(page_content="Docker volume persist", metadata={"title": "Volumes"})
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [fake_doc]

        state: RAGState = {"expanded_queries": ["query1", "query2"]}
        with patch("workflow.get_retriever", return_value=mock_retriever):
            out = retrieval(state)

        self.assertIn("raw_docs", out)
        self.assertIsInstance(out["raw_docs"], list)
        # 2 queries × 1 doc each, deduplicated if same content
        self.assertGreaterEqual(len(out["raw_docs"]), 1)
        self.assertLessEqual(len(out["raw_docs"]), 2)
        mock_retriever.invoke.assert_called()  # workers invoked


if __name__ == "__main__":
    unittest.main(verbosity=2)
