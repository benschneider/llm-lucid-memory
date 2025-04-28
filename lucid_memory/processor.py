# lucid_memory/processor.py

import os
import re
import concurrent.futures
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

# Assume these are imported correctly
from .digestor import Digestor
from .memory_graph import MemoryGraph
from .memory_node import MemoryNode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Constants ---
# Define the path relative to *this* file's location or assume it's in root?
# Let's assume it's relative to the CWD (current working directory)
# where the script (e.g., gui.py) is run from.
MEMORY_GRAPH_PATH = "memory_graph.json"
# --- End Constants ---


class ChunkProcessor:
    """
    Handles the digestion of document chunks in parallel and updates the memory graph.
    Reports progress via callback functions.
    """
    def __init__(self,
                 digestor: Digestor,
                 memory_graph: MemoryGraph,
                 status_callback: Callable[[str], None],
                 completion_callback: Callable[[bool], None]):
        # (Init remains the same)
        if not digestor: raise ValueError("Digestor instance required.")
        if not memory_graph: raise ValueError("MemoryGraph instance required.")
        self.digestor = digestor
        self.memory_graph = memory_graph
        self.status_callback = status_callback
        self.completion_callback = completion_callback

    def _digest_single_chunk_task(self, chunk_data: Dict[str, Any], original_filename: str, index: int) -> Optional[MemoryNode]:
        # (This method remains the same)
        # ... (logic to digest one chunk) ...
        try:
            chunk_content = chunk_data.get("content", "")
            chunk_metadata = chunk_data.get("metadata", {})
            chunk_id_part = chunk_metadata.get("identifier", f"chunk_{index+1}")
            sanitized_chunk_id = re.sub(r'[^\w\-]+', '_', chunk_id_part)[:50]
            base_filename_noext, _ = os.path.splitext(original_filename)
            node_id = f"file_{base_filename_noext}_{chunk_metadata.get('type','unknown')}_{sanitized_chunk_id}_{index+1}"

            logging.info(f"Attempting digestion node ID: {node_id} (Thread: {threading.current_thread().name})")
            node = self.digestor.digest(chunk_content, node_id=node_id, generate_questions=False)

            if node:
                setattr(node, 'source_chunk_metadata', chunk_metadata)
                logging.info(f"Success digest node {node_id}")
                return node
            else:
                logging.warning(f"Digest returned None chunk {index+1} (Node ID prefix: {node_id})")
                return None
        except Exception as e:
             logging.error(f"Error digest chunk {index+1} (Node ID prefix: {node_id}): {e}", exc_info=True)
             return None


    def process_chunks(self, chunks: List[Dict[str, Any]], original_filename: str):
        # (This method remains the same - uses ThreadPoolExecutor, calls _digest_single_chunk_task)
        # ... (logic to process chunks in parallel) ...
        total_chunks = len(chunks)
        if total_chunks == 0: logging.info("No chunks provided."); self.status_callback("Status: No chunks found."); self.completion_callback(False); return

        nodes_to_add: List[MemoryNode] = []
        max_workers = min(8, os.cpu_count() + 4) if os.cpu_count() else 8
        logging.info(f"Processor start parallel digest ({total_chunks} chunks, max {max_workers} workers).")
        self.status_callback(f"Status: Starting digestion ({total_chunks} chunks)...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='DigestWorker') as executor:
            future_to_chunk_index = { executor.submit(self._digest_single_chunk_task, chunk, original_filename, i): i for i, chunk in enumerate(chunks) }
            processed_chunks = 0
            for future in concurrent.futures.as_completed(future_to_chunk_index):
                chunk_index = future_to_chunk_index[future]
                try:
                    node = future.result()
                    if node: nodes_to_add.append(node)
                except Exception as exc: logging.error(f'Chunk task {chunk_index + 1} generated exception: {exc}', exc_info=True)
                processed_chunks += 1
                if processed_chunks % 5 == 0 or processed_chunks == total_chunks: self.status_callback(f"Status: Digested {processed_chunks}/{total_chunks} chunks...")

        logging.info(f"Processor finished. {len(nodes_to_add)}/{total_chunks} chunks yielded nodes.")

        graph_changed = False
        if nodes_to_add:
            logging.info(f"Adding {len(nodes_to_add)} nodes to MemoryGraph.")
            for node in nodes_to_add: self.memory_graph.add_node(node)
            graph_changed = True
            self._save_graph() # Save after adding

        final_status = f"Status: Finished {original_filename}. Added {len(nodes_to_add)}/{total_chunks} nodes."
        if len(nodes_to_add) < total_chunks: final_status += " (Check logs)"
        self.status_callback(final_status)
        self.completion_callback(graph_changed)


    def _save_graph(self):
        """Helper to save the memory graph, handling errors."""
        try:
            # Uses the MEMORY_GRAPH_PATH defined at the top of this file
            self.memory_graph.save_to_json(MEMORY_GRAPH_PATH)
            logging.info(f"Saved {len(self.memory_graph.nodes)} nodes to {MEMORY_GRAPH_PATH}")
        except Exception as e:
            logging.error(f"Memory Save Error: {e}", exc_info=True)
            self.status_callback(f"Status: Error saving memory file")