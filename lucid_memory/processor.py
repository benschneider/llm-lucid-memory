# lucid_memory/processor.py (Use Controller's _set_status for progress)

import os
import re
import concurrent.futures
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

# --- Type Hinting & Imports ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .digestor import Digestor
    from .embedder import Embedder
    from .memory_graph import MemoryGraph
    from .memory_node import MemoryNode

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Processor - %(message)s')

MEMORY_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "memory_graph.json")

class ChunkProcessor:
    """Processes chunks in parallel, updating progress via controller's callback."""
    def __init__(self,
                 digestor: 'Digestor',
                 embedder: Optional['Embedder'],
                 memory_graph: 'MemoryGraph',
                 # ---> EXPECT the Controller's _set_status method (or compatible) <---
                 status_callback: Callable[[str, Optional[int], Optional[int], Optional[str]], None],
                 # ---------------------------------------------------------------------
                 completion_callback: Callable[[bool], None]):
        """Initializes the processor."""
        if not digestor: raise ValueError("Digestor required.")
        if not memory_graph: raise ValueError("MemoryGraph required.")

        self.digestor = digestor
        self.embedder = embedder
        self.embedding_available = self.embedder is not None and self.embedder.is_available()
        self.memory_graph = memory_graph
        # Store the callback provided by the Controller
        self.controller_status_updater = status_callback # Rename for clarity
        self.controller_completion_callback = completion_callback # Rename

        logger.info(f"ChunkProcessor initialized. Embedder: {'Available' if self.embedding_available else 'Off'}")

    # --- Worker Task (_digest_and_embed_chunk_task remains the same internally) ---
    def _digest_and_embed_chunk_task(self, chunk_data: Dict[str, Any], original_filename: str, index: int) -> Optional['MemoryNode']:
        # ... (This function's internal logic for digest + embed is unchanged) ...
        # It returns Node or None
        if TYPE_CHECKING: from .memory_node import MemoryNode
        node: Optional['MemoryNode'] = None; node_id = f"ERR_ID_{index+1}"; thread_name = threading.current_thread().name
        try:
            # ... (Node ID generation) ...
            chunk_content = chunk_data.get("content", ""); chunk_metadata = chunk_data.get("metadata", {})
            chunk_id_part = chunk_metadata.get("identifier", f"anon_{index+1}"); sanitized_chunk_id = re.sub(r'[^\w\-]+', '_', chunk_id_part)[:50].strip('_') or f"chunk{index+1}"
            base_filename_noext, _ = os.path.splitext(original_filename); node_id_base = f"file_{base_filename_noext}_{chunk_metadata.get('type','txt')}_{sanitized_chunk_id}"; node_id = f"{node_id_base}_{index+1}"
            logger.debug(f"{thread_name}: Start Proc Node: {node_id}")
            # ... (Call self.digestor.digest) ...
            node = self.digestor.digest(chunk_content, node_id=node_id, chunk_metadata=chunk_metadata, generate_questions=False)
            if not node: logger.warning(f"{thread_name}: Digestor None for {node_id}. Skip."); return None
            logger.debug(f"{thread_name}: Digest OK -> {node_id}")
            # ... (Apply sequence/parent metadata) ...
            node.sequence_index = chunk_metadata.get("sequence_index"); node.parent_identifier = chunk_metadata.get("parent_identifier")
            # ... (Conditional embedding call) ...
            if self.embedding_available:
                 logger.debug(f"{thread_name}: Embedding {node_id}...")
                 embedding_vector = self.embedder.generate_embedding(node) # type: ignore
                 if embedding_vector: node.embedding = embedding_vector; logger.debug(f"{thread_name}: Embed OK {node_id} ({len(embedding_vector)}d)")
                 else: logger.warning(f"{thread_name}: Embed FAIL {node_id}.")
            else: logger.debug(f"{thread_name}: Embed SKIP {node_id}.")
            return node # Return processed node
        except Exception as e:
             log_id = getattr(node, 'id', node_id); logger.exception(f"{thread_name}: CRITICAL FAIL chunk '{log_id}': {e}"); return None


    # --- Main Processing Orchestrator ---
    def process_chunks(self, chunks: List[Dict[str, Any]], original_filename: str):
        """Executes parallel processing, calling controller's status updater."""
        total_chunks = len(chunks)
        processed_nodes: List['MemoryNode'] = []
        max_workers = max(1, min(8, (os.cpu_count() or 1) + 4))

        if total_chunks == 0:
            logger.info("Process Chunks: No input. Finished.")
            # Call status updater with message ONLY
            self.controller_status_updater(message="Status: Finished - No chunks detected.", current_step=0, total_steps=0, detail="")
            self.controller_completion_callback(False); return

        logger.info(f"Processor Starting: {total_chunks} chunks parallel (MaxW: {max_workers}). Fn='{original_filename[:30]}'")
        # Initial status update via controller's method
        self.controller_status_updater(
            message=f"Status: Starting processing ({total_chunks} chunks)...",
            current_step=0, total_steps=total_chunks, detail="Initializing..."
        )

        with concurrent.futures.ThreadPoolExecutor( max_workers=max_workers, thread_name_prefix='Lucid_Worker') as executor:
            futures_map = { executor.submit(self._digest_and_embed_chunk_task, chunk, original_filename, i): (i, chunk.get("metadata",{}).get("identifier","?")[:20]) for i, chunk in enumerate(chunks)}

            processed_count = 0; failed_count = 0
            for future in concurrent.futures.as_completed(futures_map):
                chunk_idx, chunk_disp_id = futures_map[future]
                processed_node = None
                try:
                    processed_node = future.result()
                    if processed_node: processed_nodes.append(processed_node)
                    else: failed_count += 1
                except Exception as future_exc:
                    logger.exception(f"Processor: Worker FAILED (Chunk Idx:{chunk_idx} ID:'{chunk_disp_id}') Err: {future_exc}")
                    failed_count += 1

                processed_count += 1

                # --- ** Call Controller's Status Updater with Progress ** ---
                detail_msg = f"Chunk {chunk_idx+1}: '{chunk_disp_id}'"
                detail_msg += " -> OK" if processed_node else " -> FAILED/Skipped"

                self.controller_status_updater( # <-- Call the passed method
                   message=f"Status(Bg): Processing {processed_count}/{total_chunks}",
                   current_step=processed_count,
                   total_steps=total_chunks,
                   detail=detail_msg
                )
                # ----------------------------------------------------------

        # --- Post Processing Results ---
        successful_nodes = len(processed_nodes)
        # ... (Log completion summary) ...
        final_log_msg = f"Processor JOB COMPLETE ('{original_filename}' -> {successful_nodes}/{total_chunks} succeeded nodes).";
        if failed_count > 0: final_log_msg += f" (** {failed_count} chunks FAILED/Skipped**)."; logger.info(final_log_msg)

        # --- Update Memory Graph ---
        graph_changed = False; save_ok = True
        if successful_nodes > 0:
             logger.info(f"Applying {successful_nodes} nodes to MemoryGraph..."); graph_changed = True
             for node in processed_nodes: self.memory_graph.add_node(node)
             logger.info(f"Applied {successful_nodes} nodes."); save_ok = self._save_graph();
        else: logger.info("No new nodes. Graph not modified.")

        # --- Signal FINAL Completion State via Controller Callbacks ---
        final_detail = f"{successful_nodes} nodes OK ({failed_count} errs)."
        final_status_msg = f"Status: Finished '{original_filename}'. -> {final_detail}"
        final_status_msg +=  f" | Graph Saved: {'OK' if save_ok else 'FAIL!' if graph_changed else 'No changes'}"
        # Final status update
        self.controller_status_updater(
            message=final_status_msg, current_step=total_chunks, total_steps=total_chunks, detail=final_detail
        );
        # Final completion signal
        self.controller_completion_callback(graph_changed)


    def _save_graph(self) -> bool:
        """Saves the graph using its method. Returns success bool."""
        logger.debug(f"Processor attempting graph save -> {MEMORY_GRAPH_PATH}")
        try:
           self.memory_graph.save_to_json(MEMORY_GRAPH_PATH); logger.info(f"Graph saved OK -> {MEMORY_GRAPH_PATH}"); return True
        except Exception as e:
           logger.exception(f"CRITICAL Processor FAILURE saving graph state '{MEMORY_GRAPH_PATH}': {e}")
           try: self.controller_status_updater(f"ALERT: ERROR saving Memory Graph!", None, None, None) # Simple alert msg
           except Exception: pass
           return False

# --- End of ChunkProcessor Class ---