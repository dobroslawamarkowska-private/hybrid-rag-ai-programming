"""Testy jednostkowe build_index – bez Chroma/API."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from langchain_core.documents import Document

from build_index import _chroma_safe_metadata_value, _df_to_docs


class TestChromaSafeMetadataValue(unittest.TestCase):
    """Test konwersji wartości metadanych do formatu Chroma."""

    def test_none(self):
        self.assertIsNone(_chroma_safe_metadata_value(None))

    def test_str(self):
        self.assertEqual(_chroma_safe_metadata_value("hello"), "hello")

    def test_int(self):
        self.assertEqual(_chroma_safe_metadata_value(42), 42)

    def test_float(self):
        self.assertEqual(_chroma_safe_metadata_value(3.14), 3.14)

    def test_bool(self):
        self.assertTrue(_chroma_safe_metadata_value(True))
        self.assertFalse(_chroma_safe_metadata_value(False))

    def test_list_becomes_json(self):
        val = _chroma_safe_metadata_value(["a", "b"])
        self.assertIsInstance(val, str)
        self.assertIn("a", val)
        self.assertIn("b", val)

    def test_dict_becomes_json(self):
        val = _chroma_safe_metadata_value({"k": "v"})
        self.assertIsInstance(val, str)
        self.assertIn("k", val)

    def test_numpy_array_tolist(self):
        import numpy as np

        arr = np.array([1, 2, 3])
        val = _chroma_safe_metadata_value(arr)
        self.assertIsInstance(val, str)
        self.assertIn("1", val)


class TestDfToDocs(unittest.TestCase):
    """Test konwersji DataFrame na listę Document."""

    def test_simple_row(self):
        df = pd.DataFrame([
            {"content": "Doc content", "title": "Doc Title", "file_path": "/path/doc.md"},
        ])
        docs = _df_to_docs(df)
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Doc content")
        self.assertEqual(docs[0].metadata["title"], "Doc Title")
        self.assertEqual(docs[0].metadata["file_path"], "/path/doc.md")

    def test_empty_content_uses_title_and_description(self):
        df = pd.DataFrame([
            {"content": "", "title": "T", "description": "D", "file_path": ""},
        ])
        docs = _df_to_docs(df)
        self.assertEqual(len(docs), 1)
        self.assertIn("T", docs[0].page_content)
        self.assertIn("D", docs[0].page_content)

    def test_empty_content_title_only(self):
        df = pd.DataFrame([
            {"content": "", "title": "OnlyTitle", "file_path": ""},
        ])
        docs = _df_to_docs(df)
        self.assertEqual(len(docs), 1)
        self.assertIn("OnlyTitle", docs[0].page_content)

    def test_fully_empty_rows_become_brak_tresci(self):
        df = pd.DataFrame([
            {"content": "", "title": "", "description": "", "file_path": ""},
        ])
        docs = _df_to_docs(df)
        self.assertEqual(len(docs), 1)
        self.assertIn("brak treści", docs[0].page_content)

    def test_multiple_rows(self):
        df = pd.DataFrame([
            {"content": "A", "title": "T1", "file_path": "p1"},
            {"content": "B", "title": "T2", "file_path": "p2"},
        ])
        docs = _df_to_docs(df)
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].page_content, "A")
        self.assertEqual(docs[1].page_content, "B")

    def test_optional_metadata_keys(self):
        df = pd.DataFrame([
            {"content": "X", "title": "T", "file_path": "p", "tags": ["docker"], "keywords": "docker"},
        ])
        docs = _df_to_docs(df)
        self.assertEqual(len(docs), 1)
        self.assertIn("tags", docs[0].metadata)
        self.assertIn("keywords", docs[0].metadata)


if __name__ == "__main__":
    unittest.main(verbosity=2)
