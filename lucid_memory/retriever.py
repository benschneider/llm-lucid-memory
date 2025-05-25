from collections import deque
from typing import List, Optional, Dict
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.memory_node import MemoryNode
import re
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
        query: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        max_depth: int = 1,
        max_neighbors_to_include: int = 7
    ) -> List[MemoryNode]:
        """
        Expands the context from an initial set of nodes by exploring the graph.
        Retrieves parents, sequential siblings, and nodes linked by dependency/output fields.
        """
        if not initial_nodes:
            return []

        contextual_nodes_ids = set()
        all_relevant_nodes_map = {}

        # Prioritize initial_nodes
        for node in initial_nodes:
            if node.id not in contextual_nodes_ids:
                all_relevant_nodes_map[node.id] = node
                contextual_nodes_ids.add(node.id)
                if len(all_relevant_nodes_map) >= max_neighbors_to_include:
                    break

        # If we haven't filled up max_neighbors_to_include, then explore
        nodes_to_explore_from = list(all_relevant_nodes_map.values())

        # Simple 1-hop exploration for now
        for current_node in nodes_to_explore_from: # Iterate over a copy if modifying during iteration
            if len(all_relevant_nodes_map) >= max_neighbors_to_include:
                break

            # 1. Get Parent Node (existing logic is fine)
            if current_node.parent_identifier:
                parent_node = self.memory_graph.get_node(current_node.parent_identifier)
                if parent_node and parent_node.id not in contextual_nodes_ids:
                    logger.debug(f"GraphRAG: Adding parent '{parent_node.id}' for node '{current_node.id}'")
                    all_relevant_nodes_map[parent_node.id] = parent_node
                    contextual_nodes_ids.add(parent_node.id)
                    if len(all_relevant_nodes_map) >= max_neighbors_to_include: break

            # 2. Get Sequential Siblings (existing logic is fine)
            if current_node.parent_identifier and current_node.sequence_index is not None:
                for potential_sibling in self.memory_graph.nodes.values():
                    if len(all_relevant_nodes_map) >= max_neighbors_to_include: break
                    if potential_sibling.id == current_node.id or potential_sibling.id in contextual_nodes_ids:
                        continue
                    if potential_sibling.parent_identifier == current_node.parent_identifier and \
                       potential_sibling.sequence_index is not None:
                        if abs(potential_sibling.sequence_index - current_node.sequence_index) == 1:
                            logger.debug(f"GraphRAG: Adding sibling '{potential_sibling.id}' for node '{current_node.id}'")
                            all_relevant_nodes_map[potential_sibling.id] = potential_sibling
                            contextual_nodes_ids.add(potential_sibling.id)
                if len(all_relevant_nodes_map) >= max_neighbors_to_include: break

            # 3. ADDED: Explore 'dependencies' and 'produced_outputs' as potential node links
            # This assumes these fields might contain IDs of other nodes.
            linked_node_ids_to_check = []
            if hasattr(current_node, 'dependencies') and current_node.dependencies:
                linked_node_ids_to_check.extend(current_node.dependencies)
            if hasattr(current_node, 'produced_outputs') and current_node.produced_outputs:
                linked_node_ids_to_check.extend(current_node.produced_outputs)

            for linked_id in set(linked_node_ids_to_check): # Use set to avoid processing same ID multiple times from deps/outputs
                if len(all_relevant_nodes_map) >= max_neighbors_to_include:
                    break
                if linked_id == current_node.id or linked_id in contextual_nodes_ids: # Avoid self-loops or already added
                    continue

                # Heuristic: Check if linked_id looks like a node ID we might have generated
                # (e.g., contains typical separators like '_', or is in memory_graph.nodes)
                # For now, we directly try to fetch it.
                linked_node = self.memory_graph.get_node(linked_id)
                if linked_node: # If it's a valid node ID in our graph
                    logger.debug(f"GraphRAG: Adding linked node '{linked_node.id}' (from deps/outputs) for node '{current_node.id}'")
                    all_relevant_nodes_map[linked_id] = linked_node
                    contextual_nodes_ids.add(linked_id)
            if len(all_relevant_nodes_map) >= max_neighbors_to_include: break

        final_context_nodes = list(all_relevant_nodes_map.values())

        # Refined Sorting:
        # 1. Start with initial_nodes (if they are part of the final_context_nodes)
        # 2. Then, sort other nodes perhaps by parent and sequence.
        # This ensures the directly relevant nodes from keyword search come first.

        sorted_final_nodes = []
        initial_node_ids = {n.id for n in initial_nodes}

        # Add initial nodes first, in their original reflected order (if they survived the max_neighbors cut)
        for node in initial_nodes:
            if node.id in all_relevant_nodes_map:
                sorted_final_nodes.append(all_relevant_nodes_map[node.id])

        # Add other contextual nodes, sorted structurally
        other_contextual_nodes = [
            n for n in final_context_nodes if n.id not in initial_node_ids
        ]
        other_contextual_nodes.sort(key=lambda n: (
            n.parent_identifier or "",
            n.sequence_index if n.sequence_index is not None else float('inf')
        ))
        sorted_final_nodes.extend(other_contextual_nodes)

        # De-duplicate again just in case (though map should handle it)
        # and ensure the final list is unique by ID while trying to preserve the new order
        seen_ids_for_sort = set()
        truly_final_nodes = []
        for node in sorted_final_nodes:
            if node.id not in seen_ids_for_sort:
                truly_final_nodes.append(node)
                seen_ids_for_sort.add(node.id)

        logger.info(f"GraphRAG: Expanded to {len(truly_final_nodes)} total contextual nodes from {len(initial_nodes)} initial. Query: '{query[:50]}...'")
        return truly_final_nodes[:max_neighbors_to_include]
