from typing import List, Optional
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.memory_node import MemoryNode
import re # For keyword extraction if needed later

class ReflectiveRetriever:
    def __init__(self, memory_graph: MemoryGraph):
        self.memory_graph = memory_graph

    def retrieve_by_tag(self, tag: str) -> List[MemoryNode]:
        """Retrieves nodes containing the exact tag."""
        tag = tag.lower()
        return [node for node in self.memory_graph.nodes.values() if tag in node.tags]

    def retrieve_by_keyword(self, keyword: str) -> List[MemoryNode]:
        """Retrieves nodes where keyword appears in summary, concepts, or tags."""
        keyword = keyword.lower().strip()
        if not keyword:
            return []

        results = []
        for node in self.memory_graph.nodes.values():
            # Check summary
            if keyword in node.summary.lower():
                results.append(node)
                continue # Avoid adding the same node multiple times per keyword match

            # Check key concepts (Changed from reasoning_paths)
            if any(keyword in concept.lower() for concept in node.key_concepts):
                results.append(node)
                continue

            # Check tags
            if any(keyword in tag.lower() for tag in node.tags): # Also check tags
                 results.append(node)
                 continue

        # TODO: Add embedding-based search here later for semantic similarity

        # Deduplicate results (nodes might match multiple ways)
        # Using dict keys for efficient deduplication based on node ID
        return list({node.id: node for node in results}.values())


    def reflect_on_candidates(self, candidates: List[MemoryNode], question: str) -> List[MemoryNode]:
        """
        Basic reflection: Rank candidates based on keyword matches in question.
        TODO: Enhance with LLM-based relevance scoring or graph traversal.
        """
        question_lower = question.lower()
        scored = []

        # Simple scoring based on keyword presence (can be improved)
        for node in candidates:
            score = 0
            # Score 1: Tag overlap with question
            if node.tags and any(tag in question_lower for tag in node.tags if len(tag) > 2): # Avoid tiny tags matching everywhere
                score += 1
            # Score 2: Concepts overlap with question (Changed from reasoning_paths)
            if node.key_concepts and any(concept in question_lower for concept in node.key_concepts if len(concept) > 3): # Avoid short concepts matching
                 score += 2
            # Score 3: Summary overlap (less emphasis)
            # Check for significant overlap, not just single words
            summary_words = set(re.findall(r'\b\w+\b', node.summary.lower()))
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            common_words = summary_words.intersection(question_words)
            if len(common_words) >= 2: # Require at least 2 common words for summary match
                 score += 1

            if score > 0:
                 scored.append((score, node)) # Only include nodes with some relevance

        # Sort by score descending
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top N candidates or all relevant ones? For now, return all scored.
        # Limit could be added: return [node for score, node in scored[:5]]
        return [node for score, node in scored]