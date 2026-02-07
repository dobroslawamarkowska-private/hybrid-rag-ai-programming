"""Test suites for Docker docs retriever - simple, complex tech, and semantic queries."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever import get_retriever, create_docker_docs_tool

# Testy integracyjne (wymagają indeksu Chroma i API) – można pominąć: SKIP_INTEGRATION=1
SKIP_INTEGRATION = os.environ.get("SKIP_INTEGRATION", "").lower() in ("1", "true", "yes")


def _print_results(query: str, results: list, suite_name: str, query_num: int) -> None:
    """Nicely print query and top results."""
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  [{suite_name}] Query {query_num}: {query}")
    print(sep)
    for i, doc in enumerate(results[:3], 1):
        content = doc.page_content.strip()
        preview = content[:300] + "..." if len(content) > 300 else content
        meta = doc.metadata
        title = meta.get("title", meta.get("file_path", "—"))
        print(f"\n  📄 Result {i} (title: {title})")
        print(f"  {preview}\n")


# --- Suite 1: Simple cases ---
SIMPLE_QUERIES = [
    "How to build a Docker image?",
    "What is Dockerfile?",
    "How to run a container?",
]

# --- Suite 2: Complex tech questions ---
COMPLEX_QUERIES = [
    "How to expose ports in Docker?",
    "Difference between CMD and ENTRYPOINT.",
    "How to use docker compose?",
]

# --- Suite 3: Semantic questions ---
SEMANTIC_QUERIES = [
    "How can I persist data in Docker containers?",
    "How to debug a failing container?",
    "How to reduce Docker image size?",
]


@unittest.skipIf(SKIP_INTEGRATION, "SKIP_INTEGRATION=1")
class TestRetrieverStructure(unittest.TestCase):
    """Struktura toola i retrievera (wymaga indeksu Chroma)."""

    def test_create_docker_docs_tool_returns_tool(self):
        tool = create_docker_docs_tool()
        self.assertIsNotNone(tool)
        self.assertEqual(tool.name, "search_docker_docs")
        self.assertIn("Docker", tool.description)

    def test_get_retriever_respects_k(self):
        retriever = get_retriever(k=2)
        results = retriever.invoke("Docker")
        self.assertLessEqual(len(results), 2)


@unittest.skipIf(SKIP_INTEGRATION, "SKIP_INTEGRATION=1 – pomijam testy integracyjne")
class TestSimpleQueries(unittest.TestCase):
    """Simple, direct questions about Docker basics."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = get_retriever()

    def test_how_to_build_image(self):
        query = SIMPLE_QUERIES[0]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Simple", 1)

    def test_what_is_dockerfile(self):
        query = SIMPLE_QUERIES[1]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Simple", 2)

    def test_how_to_run_container(self):
        query = SIMPLE_QUERIES[2]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Simple", 3)


@unittest.skipIf(SKIP_INTEGRATION, "SKIP_INTEGRATION=1")
class TestComplexTechQueries(unittest.TestCase):
    """More complex technical Docker questions."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = get_retriever()

    def test_expose_ports(self):
        query = COMPLEX_QUERIES[0]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Complex", 1)

    def test_cmd_vs_entrypoint(self):
        query = COMPLEX_QUERIES[1]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Complex", 2)

    def test_docker_compose(self):
        query = COMPLEX_QUERIES[2]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Complex", 3)


@unittest.skipIf(SKIP_INTEGRATION, "SKIP_INTEGRATION=1")
class TestSemanticQueries(unittest.TestCase):
    """Semantic/conceptual questions (require understanding intent)."""

    @classmethod
    def setUpClass(cls):
        cls.retriever = get_retriever()

    def test_persist_data(self):
        query = SEMANTIC_QUERIES[0]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Semantic", 1)

    def test_debug_failing_container(self):
        query = SEMANTIC_QUERIES[1]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Semantic", 2)

    def test_reduce_image_size(self):
        query = SEMANTIC_QUERIES[2]
        results = self.retriever.invoke(query)
        self.assertGreater(len(results), 0, f"Query returned no results: {query!r}")
        _print_results(query, results, "Semantic", 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
