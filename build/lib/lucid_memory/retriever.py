

from typing import List, Optional
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.memory_node import MemoryNode

class ReflectiveRetriever:
    def __init__(self, memory_graph: MemoryGraph):
        self.memory_graph = memory_graph

    def retrieve_by_tag(self, tag: str) -> List[MemoryNode]:
        return self.memory_graph.search_by_tag(tag)

    def retrieve_by_keyword(self, keyword: str) -> List[MemoryNode]:
        keyword = keyword.lower()
        return [
            node for node in self.memory_graph.nodes.values()
            if keyword in node.summary.lower() or any(keyword in rp.lower() for rp in node.reasoning_paths)
        ]

    def reflect_on_candidates(self, candidates: List[MemoryNode], question: str) -> List[MemoryNode]:
        """Rank and filter candidates based on how well they seem to match the question."""
        question = question.lower()
        scored = []
        for node in candidates:
            score = 0
            if any(word in question for word in node.tags):
                score += 2
            if node.summary.lower() in question:
                score += 3
            if any(rp.lower() in question for rp in node.reasoning_paths):
                score += 1
            scored.append((score, node))
        scored.sort(reverse=True, key=lambda x: x[0])
        return [node for score, node in scored if score > 0]