import os
import re
import concurrent.futures
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .digestor import Digestor
    from .embedder import Embedder
    from .memory_graph import MemoryGraph
    from .memory_node import MemoryNode

logger = logging.getLogger(__name__) # Use module-level logger

LUCID_MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEMORY_GRAPH_PATH_PROC = os.path.join(LUCID_MEMORY_DIR, "memory_graph.json")


class ChunkProcessor:
    """Processes chunks in parallel, updating progress via controller's callback."""
    def __init__(self,
                 digestor: 'Digestor',
                 embedder: Optional['Embedder'],
                 memory_graph: 'MemoryGraph',
                 status_callback: Callable[[str, Optional[int], Optional[int], Optional[str]], None],
                 completion_callback: Callable[[bool], None],
                 memory_graph_path_override: Optional[str] = None): # ADDED override
        if not digestor: raise ValueError("Digestor instance is required for ChunkProcessor.")
        if not memory_graph: raise ValueError("MemoryGraph instance is required for ChunkProcessor.")

        self.digestor = digestor
        self.embedder = embedder
        self.embedding_available = self.embedder is not None and self.embedder.is_available()
        self.memory_graph = memory_graph # This is the graph instance to update
        
        # Use override if provided, otherwise use the default path
        self.memory_graph_file_path = memory_graph_path_override or DEFAULT_MEMORY_GRAPH_PATH_PROC

        self.controller_status_updater = status_callback
        self.controller_completion_callback = completion_callback

        logger.info(f"ChunkProcessor initialized. Embedder: {'Available' if self.embedding_available else 'Not Available/Configured'}. Graph Path: {self.memory_graph_file_path}")

    def _digest_and_embed_chunk_task(self, chunk_data: Dict[str, Any], original_filename: str, index: int) -> Optional['MemoryNode']:
        if TYPE_CHECKING: from .memory_node import MemoryNode # For linter
        
        node: Optional['MemoryNode'] = None
        node_id = f"error_node_id_{index+1}" # Fallback ID
        thread_name = threading.current_thread().name
        
        try:
            chunk_content = chunk_data.get("content", "")
            chunk_metadata = chunk_data.get("metadata", {})
            
            if not chunk_content.strip():
                logger.warning(f"{thread_name}: Skipping empty chunk at index {index} for '{original_filename}'.")
                return None

            # Generate a more robust node_id
            chunk_type = chunk_metadata.get("type", "unknown").replace("python_", "py_") # Shorten "python_"
            chunk_identifier = chunk_metadata.get("identifier", f"chunk_{index+1}")
            
            # Sanitize parts of the ID
            sanitized_filename = re.sub(r'[^\w\-.]+', '_', os.path.splitext(original_filename)[0])[:50]
            sanitized_chunk_id_part = re.sub(r'[^\w\-]+', '_', chunk_identifier)[:50].strip('_') or f"part{index+1}"
            
            node_id = f"{sanitized_filename}_{chunk_type}_{sanitized_chunk_id_part}"
            # Ensure uniqueness if multiple chunks have same identifier (e.g. paragraphs)
            # Check if this ID already exists (if processing multiple files into one graph, could happen)
            # For now, assume IDs from one file processing run are unique due to content/structure.
            # If adding an index, node_id = f"{node_id_base}_{index}"

            logger.debug(f"{thread_name}: Starting processing for Node ID tentative: {node_id} (Index: {index})")
            
            node = self.digestor.digest(
                raw_text=chunk_content, 
                node_id=node_id, # Pass the generated ID
                chunk_metadata=chunk_metadata, 
                generate_questions=False # Configurable?
            )

            if not node:
                logger.warning(f"{thread_name}: Digestor returned None for chunk (orig_id_part: {chunk_identifier}, index: {index}). Skipping.")
                return None
            
            # Ensure the node's ID is what we expect, or update if digestor changes it (shouldn't ideally)
            node.id = node_id # Enforce the ID we generated

            logger.debug(f"{thread_name}: Digestion successful for Node ID: {node.id}")
            
            # Add structural metadata from chunker to the node
            node.sequence_index = chunk_metadata.get("sequence_index")
            node.parent_identifier = chunk_metadata.get("parent_identifier")
            node.source = original_filename # Store original file name

            if self.embedding_available and self.embedder: # Check self.embedder too
                 logger.debug(f"{thread_name}: Attempting to generate embedding for Node ID: {node.id}...")
                 embedding_vector = self.embedder.generate_embedding(node)
                 if embedding_vector:
                     node.embedding = embedding_vector # Store embedding on the node
                     logger.debug(f"{thread_name}: Embedding successful for Node ID: {node.id} (Dim: {len(embedding_vector)})")
                 else:
                     logger.warning(f"{thread_name}: Embedding generation FAILED for Node ID: {node.id}.")
            else:
                 logger.debug(f"{thread_name}: Embedding skipped for Node ID: {node.id} (Embedder not available or not configured).")
            
            return node
        except Exception as e:
             current_node_id_for_log = getattr(node, 'id', node_id) # Use node's actual ID if available
             logger.exception(f"{thread_name}: CRITICAL FAILURE processing chunk for Node ID '{current_node_id_for_log}' (Index: {index}): {e}")
             return None


    def process_chunks(self, chunks: List[Dict[str, Any]], original_filename: str) -> None: # MODIFIED: returns None
        """
        Executes parallel processing of chunks. Updates are sent via status_callback.
        Saves the memory_graph upon completion if changes were made.
        This method does NOT return the nodes directly; it updates the provided memory_graph instance.
        """
        total_chunks = len(chunks)
        if total_chunks == 0:
            logger.info(f"Process Chunks: No input chunks for '{original_filename}'. Task finished.")
            self.controller_status_updater(message=f"Status: Finished '{original_filename}' - No chunks to process.", current_step=0, total_steps=0, detail="")
            self.controller_completion_callback(False); return

        max_workers = max(1, min(8, (os.cpu_count() or 1) + 2)) # Adjusted max_workers
        logger.info(f"ChunkProcessor starting job for '{original_filename}': {total_chunks} chunks using up to {max_workers} workers.")
        self.controller_status_updater(
            message=f"Status: Starting processing for '{original_filename}' ({total_chunks} chunks)...",
            current_step=0, total_steps=total_chunks, detail="Initializing workers..."
        )

        processed_nodes_for_graph: List['MemoryNode'] = [] # Collect successfully processed nodes here
        failed_count = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Lucid_Worker') as executor:
            # Map future to (index, chunk_display_identifier) for better error reporting/status
            futures_map = {
                executor.submit(self._digest_and_embed_chunk_task, chunk, original_filename, i): 
                (i, chunk.get("metadata",{}).get("identifier","<unknown_id>")[:30]) 
                for i, chunk in enumerate(chunks)
            }

            processed_count = 0
            for future in concurrent.futures.as_completed(futures_map):
                chunk_idx, chunk_disp_id = futures_map[future]
                processed_node_result: Optional['MemoryNode'] = None
                try:
                    processed_node_result = future.result()
                    if processed_node_result:
                        processed_nodes_for_graph.append(processed_node_result)
                    else:
                        failed_count += 1
                except Exception as future_exc:
                    logger.error(f"Processor: Worker thread FAILED for chunk (Index: {chunk_idx}, ID Hint: '{chunk_disp_id}') - Error: {future_exc}", exc_info=True)
                    failed_count += 1
                
                processed_count += 1
                detail_msg = f"Chunk {processed_count} ({chunk_disp_id})"
                detail_msg += " -> Processed OK" if processed_node_result else " -> FAILED/Skipped"
                self.controller_status_updater(
                   message=f"Status(Bg): Processing {processed_count}/{total_chunks} for '{original_filename}'",
                   current_step=processed_count,
                   total_steps=total_chunks,
                   detail=detail_msg
                )

        successful_nodes_count = len(processed_nodes_for_graph)
        final_log_msg = f"ChunkProcessor JOB COMPLETE for '{original_filename}': {successful_nodes_count}/{total_chunks} nodes successfully processed."
        if failed_count > 0: final_log_msg += f" ({failed_count} chunks FAILED or were skipped)."
        logger.info(final_log_msg)

        graph_changed_in_this_run = False
        save_successful = True # Assume true unless a save fails

        if successful_nodes_count > 0:
             logger.info(f"Applying {successful_nodes_count} new/updated nodes to the MemoryGraph instance...")
             for node_to_add in processed_nodes_for_graph:
                 self.memory_graph.add_node(node_to_add) # Update the graph instance
             graph_changed_in_this_run = True
             logger.info(f"MemoryGraph instance updated with {successful_nodes_count} nodes. Attempting to save graph to file: {self.memory_graph_file_path}")
             save_successful = self._save_graph() # Save the updated graph
        else:
             logger.info("No new nodes were successfully processed. MemoryGraph instance not modified by this run.")

        final_detail_for_ui = f"{successful_nodes_count} nodes processed ({failed_count} errors/skips)."
        final_status_msg_for_ui = f"Status: Finished processing '{original_filename}'. {final_detail_for_ui}"
        if graph_changed_in_this_run:
            final_status_msg_for_ui += f" | Graph saved: {'OK' if save_successful else 'FAILED!'}"
        
        self.controller_status_updater(
            message=final_status_msg_for_ui, current_step=total_chunks, total_steps=total_chunks, detail=final_detail_for_ui
        )
        self.controller_completion_callback(graph_changed_in_this_run and save_successful)


    def _save_graph(self) -> bool:
        """Saves the memory_graph instance to the configured file path. Returns success bool."""
        logger.debug(f"Processor attempting to save memory graph to: {self.memory_graph_file_path}")
        try:
           # The memory_graph instance itself has the nodes, save it.
           self.memory_graph.save_to_json(self.memory_graph_file_path)
           logger.info(f"Memory graph successfully saved to {self.memory_graph_file_path}")
           return True
        except Exception as e:
           logger.exception(f"CRITICAL Processor FAILURE: Could not save memory graph to '{self.memory_graph_file_path}': {e}")
           # Notify controller about save failure through status if possible (though it might be too late for detailed progress)
           try:
               self.controller_status_updater(f"ALERT: CRITICAL ERROR saving Memory Graph to {self.memory_graph_file_path}!", None, None, "Graph Save Failed")
           except Exception: pass # Avoid errors in error reporting
           return False