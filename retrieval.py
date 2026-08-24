"""Root re-export for retrieval.py"""
import sys
from pathlib import Path

src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from knowledge_analyst.retrieval import vector_search, web_search, SEED_DOCUMENTS

__all__ = ["vector_search", "web_search", "SEED_DOCUMENTS"]

if __name__ == "__main__":
    v = vector_search("fintech Nigeria")
    print(f"Vector search returned {len(v)} results")
    w = web_search("AI Africa adoption")
    print(f"Web search returned {len(w)} results")
