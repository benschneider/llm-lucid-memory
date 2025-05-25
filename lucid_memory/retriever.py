from collections import deque
from typing import List, Optional
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.memory_node import MemoryNode
import re  # For keyword extraction if needed later
import logging

logger = logging.getLogger(__name__)

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
                continue

            # Check key concepts (Changed from reasoning_paths)
            if any(keyword in concept.lower() for concept in node.key_concepts):
                results.append(node)
                continue

            # Check tags
            if any(keyword in tag.lower() for tag in node.tags):
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
            if node.tags and any(tag in question_lower for tag in node.tags if len(tag) > 2):  # Avoid tiny tags matching everywhere
                score += 1
            # Score 2: Concepts overlap with question (Changed from reasoning_paths)
            if node.key_concepts and any(concept in question_lower for concept in node.key_concepts if len(concept) > 3):  # Avoid short concepts matching
                score += 2
            # Score 3: Summary overlap (less emphasis)
            # Check for significant overlap, not just single words
            summary_words = set(re.findall(r'\b\w+\b', node.summary.lower()))
            question_words = set(re.findall(r'\b\w+\b', question_lower))
            common_words = summary_words.intersection(question_words)
            if len(common_words) >= 2:  # Require at least 2 common words for summary match
                score += 1

            if score > 0:
                scored.append((score, node))  # Only include nodes with some relevance

        # Sort by score descending
        scored.sort(reverse=True, key=lambda x: x[0])

        # Return top N candidates or all relevant ones? For now, return all scored.
        # Limit could be added: return [node for score, node in scored[:5]]
        return [node for score, node in scored]

    def retrieve_graph_context(
        self,
        initial_nodes: List[MemoryNode],
        query: str,  # Keep query for potential future relevance scoring of neighbors
        max_depth: int = 1,  # How many steps to explore (1 means direct parent/siblings)
        max_neighbors_to_include: int = 5  # Max total nodes to return including initial
    ) -> List[MemoryNode]:
        """
        Expands the context from an initial set of nodes by exploring the graph.
        Retrieves parents and sequential siblings of the initial nodes.

        Args:
            initial_nodes: A list of MemoryNode objects that are initially relevant.
            query: The user's query string (for potential future use in ranking neighbors).
            max_depth: Not fully implemented for complex traversal yet, conceptual.
            max_neighbors_to_include: Limits the total number of nodes returned.

        Returns:
            A de-duplicated list of MemoryNode objects including initial and context nodes,
            potentially re-sorted or prioritized.
        """
        if not initial_nodes:
            return []

        # Use a set to keep track of node IDs to ensure uniqueness and avoid cycles
        contextual_nodes_ids = set(node.id for node in initial_nodes)
        # Use a list to maintain some order, starting with initial nodes
        # (Order might be lost due to set operations later, or re-established by sorting)
        all_relevant_nodes_map = {node.id: node for node in initial_nodes}

        # For each initial node, find its parent and direct sequential siblings
        nodes_to_explore_further = list(initial_nodes)  # Start with initial nodes

        # Simple 1-hop exploration for now
        for current_node in nodes_to_explore_further:
            if len(all_relevant_nodes_map) >= max_neighbors_to_include:
                break  # Stop if we've gathered enough context

            # 1. Get Parent Node
            if current_node.parent_identifier:
                parent_node = self.memory_graph.get_node(current_node.parent_identifier)
                if parent_node and parent_node.id not in contextual_nodes_ids:
                    logger.debug(f"GraphRAG: Adding parent '{parent_node.id}' for node '{current_node.id}'")
                    all_relevant_nodes_map[parent_node.id] = parent_node
                    contextual_nodes_ids.add(parent_node.id)
                    # Do not add parent to nodes_to_explore_further to avoid deep dives in this simple version

            if len(all_relevant_nodes_map) >= max_neighbors_to_include:
                break

            # 2. Get Sequential Siblings (nodes with same parent and adjacent sequence_index)
            if current_node.parent_identifier and current_node.sequence_index is not None:
                # Iterate through all nodes to find siblings - can be optimized if graph stores parent-child explicitly
                for potential_sibling in self.memory_graph.nodes.values():
                    if len(all_relevant_nodes_map) >= max_neighbors_to_include:
                        break
                    if potential_sibling.id == current_node.id:
                        continue  # Skip self

                    if potential_sibling.parent_identifier == current_node.parent_identifier and \
                       potential_sibling.sequence_index is not None:
                        # Check for previous sibling
                        if potential_sibling.sequence_index == current_node.sequence_index - 1 and \
                           potential_sibling.id not in contextual_nodes_ids:
                            logger.debug(f"GraphRAG: Adding prev sibling '{potential_sibling.id}' for node '{current_node.id}'")
                            all_relevant_nodes_map[potential_sibling.id] = potential_sibling
                            contextual_nodes_ids.add(potential_sibling.id)
                            # Do not add siblings to nodes_to_explore_further in this simple version

                        # Check for next sibling
                        if potential_sibling.sequence_index == current_node.sequence_index + 1 and \
                           potential_sibling.id not in contextual_nodes_ids:
                            logger.debug(f"GraphRAG: Adding next sibling '{potential_sibling.id}' for node '{current_node.id}'")
                            all_relevant_nodes_map[potential_sibling.id] = potential_sibling
                            contextual_nodes_ids.add(potential_sibling.id)
                            # Do not add siblings to nodes_to_explore_further

        # Convert map back to list
        final_context_nodes = list(all_relevant_nodes_map.values())

        # Optional: Re-sort the nodes. A simple sort could be by original file sequence if available.
        # For now, the order will be somewhat arbitrary based on discovery.
        # A more sophisticated approach would rank them by relevance to the query or structural importance.
        # Example sort by sequence_index (if all from same parent, or if sequence is global)
        # This assumes sequence_index is globally meaningful or nodes are primarily from one context.
        final_context_nodes.sort(key=lambda n: (
            n.parent_identifier or "",  # Group by parent first
            n.sequence_index if n.sequence_index is not None else float('inf')  # Then by sequence
        ))

        logger.info(f"GraphRAG: Expanded {len(initial_nodes)} initial nodes to {len(final_context_nodes)} total contextual nodes.")
        return final_context_nodes[:max_neighbors_to_include]  # Ensure max limit
