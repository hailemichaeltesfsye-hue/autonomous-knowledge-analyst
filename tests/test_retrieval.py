"""
Unit tests for multi-strategy retrieval engine (Vector & Web search).
"""

import sys
from pathlib import Path

# Add src path
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from knowledge_analyst.retrieval import vector_search, web_search, SEED_DOCUMENTS


def test_vector_search_returns_snippets():
    results = vector_search("fintech credit scoring Nigeria", top_k=2)
    assert isinstance(results, list)
    assert len(results) > 0
    assert any("fintech" in r.lower() or "nigeria" in r.lower() or "scoring" in r.lower() for r in results)


def test_web_search_returns_snippets():
    results = web_search("AI adoption African businesses 2024", max_results=2)
    assert isinstance(results, list)
    assert len(results) > 0


def test_seed_documents_count():
    assert len(SEED_DOCUMENTS) == 15


if __name__ == "__main__":
    print("Running test_vector_search_returns_snippets...")
    test_vector_search_returns_snippets()
    print("Running test_web_search_returns_snippets...")
    test_web_search_returns_snippets()
    print("Running test_seed_documents_count...")
    test_seed_documents_count()
    print("All retrieval tests passed successfully!")
