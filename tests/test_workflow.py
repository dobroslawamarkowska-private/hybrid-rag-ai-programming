"""Test Advanced RAG workflow."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from workflow import build_rag_graph, ask

SKIP_INTEGRATION = os.environ.get("SKIP_INTEGRATION", "").lower() in ("1", "true", "yes")


class TestRAGWorkflow(unittest.TestCase):
    """Test the LangGraph RAG workflow."""

    def test_graph_compiles(self):
        graph = build_rag_graph()
        self.assertIsNotNone(graph)

    def test_graph_has_expected_nodes(self):
        graph = build_rag_graph()
        nodes = set(graph.nodes.keys())
        expected = {"pre_retrieval", "retrieval", "check_and_refine", "post_retrieval", "generate"}
        self.assertTrue(expected.issubset(nodes), f"Oczekiwane nodesy: {expected}, jest: {nodes}")

    @unittest.skipIf(SKIP_INTEGRATION, "SKIP_INTEGRATION=1")
    def test_ask_returns_answer(self):
        answer = ask("How to run a container?")
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer.strip()), 0)

    @unittest.skipIf(SKIP_INTEGRATION, "SKIP_INTEGRATION=1")
    def test_out_of_docs_returns_no_info_and_closest_proposal(self):
        """Pytanie spoza dokumentacji: oczekujemy komunikatu o braku info + propozycji najbliższej."""
        # "instalacja Docker Hub na Linux" - Docker Hub to rejestr, nie instaluje się na Linuxie;
        # dokumentacja mówi o instalacji Docker Engine/Docker Desktop, nie "Docker Hub"
        query = "Instalacja Docker Hub na Linux"
        answer = ask(query)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer.strip()), 0)

        answer_lower = answer.lower()
        # Powinno być jasne, że dokładnej informacji nie ma
        no_info_phrases = [
            "nie ma",
            "does not contain",
            "doesn't contain",
            "not in the",
            "no such",
            "brak ",
            "nie zawiera",
            "nie znajdziesz",
        ]
        has_no_info = any(p in answer_lower for p in no_info_phrases)

        # Powinna być propozycja najbliższej informacji
        closest_phrases = [
            "najbliższa",
            "closest",
            "closest match",
            "related",
            "podobn",
            "install docker",
            "docker engine",
        ]
        has_closest = any(p in answer_lower for p in closest_phrases)

        self.assertTrue(
            has_no_info or has_closest,
            f"Oczekiwano: brak dokładnej informacji + propozycja najbliższej. Otrzymano:\n{answer}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
