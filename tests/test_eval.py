"""Testy evaluatorów ewaluacji RAG (eval_rag.py)."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_rag import (
    answer_not_empty,
    expected_keywords_present,
    qa_correctness,
    predict,
)


class TestAnswerNotEmpty(unittest.TestCase):
    """Test evaluatora answer_not_empty."""

    def test_non_empty_returns_1(self):
        run = MagicMock(outputs={"answer": "Some answer"})
        example = MagicMock()
        result = answer_not_empty(run, example)
        self.assertEqual(result.score, 1.0)

    def test_empty_returns_0(self):
        run = MagicMock(outputs={"answer": ""})
        example = MagicMock()
        result = answer_not_empty(run, example)
        self.assertEqual(result.score, 0.0)

    def test_whitespace_only_returns_0(self):
        run = MagicMock(outputs={"answer": "   \n"})
        example = MagicMock()
        result = answer_not_empty(run, example)
        self.assertEqual(result.score, 0.0)


class TestExpectedKeywordsPresent(unittest.TestCase):
    """Test evaluatora expected_keywords_present."""

    def test_all_keywords_found(self):
        run = MagicMock(outputs={"answer": "Use docker volume and bind mount for data"})
        example = MagicMock(outputs={"expected_keywords": ["volume", "bind", "mount"]})
        result = expected_keywords_present(run, example)
        self.assertEqual(result.score, 1.0)
        self.assertIn("3/3", result.comment)

    def test_partial_keywords(self):
        run = MagicMock(outputs={"answer": "docker volume for storage"})
        example = MagicMock(outputs={"expected_keywords": ["volume", "bind", "mount"]})
        result = expected_keywords_present(run, example)
        self.assertAlmostEqual(result.score, 1 / 3, places=2)

    def test_no_keywords_returns_1(self):
        run = MagicMock(outputs={"answer": "anything"})
        example = MagicMock(outputs={"expected_keywords": []})
        result = expected_keywords_present(run, example)
        self.assertEqual(result.score, 1.0)
        self.assertIn("No keywords", result.comment)


class TestQaCorrectness(unittest.TestCase):
    """Test evaluatora qa_correctness (LLM-as-judge)."""

    def test_skip_when_no_expected_answer(self):
        run = MagicMock(outputs={"answer": "foo"})
        example = MagicMock(outputs={})
        result = qa_correctness(run, example)
        self.assertEqual(result.score, 1.0)
        self.assertIn("Skipped", result.comment)

    def test_invokes_llm_when_expected_answer(self):
        run = MagicMock(outputs={"answer": "Use docker volume"}, inputs={"query": "How to persist?"})
        example = MagicMock(
            outputs={"expected_answer": "Use volumes"},
            inputs={"query": "How to persist?"},
        )
        with patch("eval_rag._get_eval_llm") as mock_llm:
            mock_grader = MagicMock()
            mock_grader.invoke.return_value = MagicMock(score=0.9, comment="Good")
            mock_llm.return_value.with_structured_output.return_value = mock_grader
            result = qa_correctness(run, example)
        self.assertEqual(result.score, 0.9)
        self.assertEqual(result.comment, "Good")


class TestPredict(unittest.TestCase):
    """Test funkcji predict (target dla evaluate)."""

    def test_predict_returns_dict_with_answer(self):
        with patch("eval_rag.ask", return_value="Docker volumes persist data"):
            out = predict({"query": "How to persist?"})
        self.assertEqual(out["answer"], "Docker volumes persist data")

    def test_predict_empty_query(self):
        with patch("eval_rag.ask", return_value=""):
            out = predict({"query": ""})
        self.assertEqual(out["answer"], "")


if __name__ == "__main__":
    unittest.main()
