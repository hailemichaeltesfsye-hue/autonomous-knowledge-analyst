"""Root re-export for graph.py"""
import sys
from pathlib import Path

# Add src to python path for package imports
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from knowledge_analyst.graph import (
    create_graph,
    get_checkpointer,
    route_after_judge,
    route_after_hitl
)

__all__ = [
    "create_graph",
    "get_checkpointer",
    "route_after_judge",
    "route_after_hitl"
]

if __name__ == "__main__":
    app = create_graph()
    print("Graph initialized from root graph.py:")
    print(app)
