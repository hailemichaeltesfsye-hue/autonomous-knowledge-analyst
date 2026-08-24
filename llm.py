"""Root re-export for llm.py"""
import sys
from pathlib import Path

src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from knowledge_analyst.llm import get_llm, DEFAULT_MODEL

__all__ = ["get_llm", "DEFAULT_MODEL"]
