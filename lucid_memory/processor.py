# FILE: lucid_memory/processor.py

import os
import re
import concurrent.futures
import logging
import threading
from typing import List, Dict, Any, Optional, Callable

# Use TYPE_CHECKING for component imports if needed to avoid cycles
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .digestor import Digestor
    from .embedder import Embedder
    from .memory_graph import MemoryGraph
    from .memory_node import MemoryNode

# --- Logging Setup ---
# Consider making logger name specific to module
logger = logging.getLogger(__name__)
# Inherit root logger settings (set in controller/app?) or configure here:
if not logger.hasHandlers(): # Configure only if not already configured by parent
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Processor - %(message)s')

# --- Constants ---
# Get Memory Graph path from external source? Ok keep simple defined path for save..
MEMORY_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "memory_graph.json")

class ChunkProcessor:
    """
    Handles parallel processing of chunks: digestion by Digestor and
    embedding generation by Embedder. Updates MemoryGraph.
    """
    # **** THIS IS THE METHOD TO FIX ****
    def __init__(self,
                 # Required components injected
                 digestor: 'Digestor', # Use string ref for Digestor
                 embedder: Optional['Embedder'], # <<< ADDED embedder parameter here
                 memory_graph: 'MemoryGraph', # Use string Ref for MemoryGraph
                 # Callbacks
                 status_callback: Callable[[str], None],
                 completion_callback: Callable[[bool], None]):
        """
        Initializes the processor with necessary components.

        Args:
            digestor: The LLM processing component. Required.
            embedder: The vector embedding component. Optional.
            memory_graph: Graph instance to update. Required.
            status_callback: Function to send status messages to.
            completion_callback: Function call when processing finishes.
        """
        if not digestor: raise ValueError("Processor requires a valid Digestor instance.")
        if not memory_graph: raise ValueError("Processor requires a valid MemoryGraph instance.")

        self.digestor = digestor
        # --- Store the passed embedder ---
        self.embedder = embedder
        # --- Check its readiness **immediately** ---
        self.embedding_available = self.embedder is not None and self.embedder.is_available()
        # -----------------------------------
        self.memory_graph = memory_graph
        self.status_callback = status_callback
        self.completion_callback = completion_callback

        logger.info(f"ChunkProcessor initialized. Digestor: Ready, Embedder: {'Available' if self.embedding_available else 'Unavailable/Off'}")
    # **** END FIX AREA ****

    # --- Worker Task: Digest AND Embed ---
    def _digest_and_embed_chunk_task(self, chunk_data: Dict[str, Any], original_filename: str, index: int) -> Optional['MemoryNode']:
        """WORKER TASK: Processes a single chunk (digest + embed)."""
        # --- Need MemoryNode IMPORT available here --> Add TYPE_CHECKING pattern for Hinting ok... or Local import.. Assume TYPE_CHECKING works ok..
        if TYPE_CHECKING: from .memory_node import MemoryNode # Relative import inside function OK also pattern..

        node: Optional['MemoryNode'] = None # Type hint OK..
        chunk_metadata = chunk_data.get("metadata", {})
        node_id = f"PROCESSING_ERROR_ID_{index+1}" # Default Error ID useful trace..
        thread_name = threading.current_thread().name

        try:
             # --- Node ID Generation (Keep safe robust logic) ---
            chunk_content = chunk_data.get("content", "")
            chunk_id_part = chunk_metadata.get("identifier", f"anon_{index+1}")
            sanitized_chunk_id = re.sub(r'[^\w\-]+', '_', chunk_id_part)[:50].strip('_') or f"chunk{index+1}"
            base_filename_noext, _ = os.path.splitext(original_filename)
            node_id_base = f"file_{base_filename_noext}_{chunk_metadata.get('type','txt')}_{sanitized_chunk_id}" # default type TXT ok..
            node_id = f"{node_id_base}_{index+1}" # Final ID uses index ensure uniqueness.

            logger.debug(f"{thread_name}: Starting process for Node: {node_id}")

             # --- 1. Digestion ---
            node = self.digestor.digest( # Instance variable -> ok..
                chunk_content,
                node_id=node_id,
                chunk_metadata=chunk_metadata,
                generate_questions=False # Keep false for now
             )

            if not node: # Digestor failed to return Node object.. BAD error state maybe..
                 logger.warning(f"{thread_name}: Digestor returned None for chunk index {index+1} (ID: {node_id}). Chunk skipped.")
                 return None # Cannot proceed without node object..

            logger.debug(f"{thread_name}: Digestion SUCCESS -> Node: {node_id}")

             # Apply linking metadata AFTER node creation
            node.sequence_index = chunk_metadata.get("sequence_index")
            node.parent_identifier = chunk_metadata.get("parent_identifier")
            # setattr(node, 'source_chunk_metadata', chunk_metadata) # Optional store Raw meta too? skip now..

            # --- 2. Embedding (Conditional) ---
            if self.embedding_available: # Check Stored instance Flag OK-> Checked in __init__ ok..
                 logger.debug(f"{thread_name}: Attempting Embedding Node: {node_id}")
                 embedding_vector = self.embedder.generate_embedding(node) # Call the OTHER component..

                 if embedding_vector:
                     node.embedding = embedding_vector # --> Attach the VECTOR to node object!
                     logger.debug(f"{thread_name}: Embedding SUCCESS Node: {node_id} (Dim: {len(embedding_vector)})")
                 else:
                     logger.warning(f"{thread_name}: Embedding FAILED or NO vector Node: {node_id}. Stored without embedding.")
            else:
                logger.debug(f"{thread_name}: Embedding SKIPPED (unavailable/off) Node: {node_id}.")

            return node # Return completed node (with or without embedding)

        except Exception as e:
             # Node ID might be constructed or default Error ID.
             log_id = getattr(node, 'id', node_id ) # Safely get ID For logging error Message.. nice pattern.. ok..
             logger.exception(f"{thread_name}: CRITICAL FAILURE processing chunk '{log_id}': {e}")
             return None # Indicate failure for this chunk task.


    # --- Main Processing Orchestrator ---
    def process_chunks(self, chunks: List[Dict[str, Any]], original_filename: str):
         """Executes the parallel digestion and embedding for list of chunks."""
         total_chunks = len(chunks)
         processed_nodes: List['MemoryNode'] = [] # Explicit type hint Optional? No LIST needed.
         # Calculate max worker count safer default=MIN 1 --> MAX (usually 8) ok..
         max_workers = max(1, min(8, (os.cpu_count() or 1) + 4)) # Use safe 'or 1' CPU unknown state.. ok..

         if total_chunks == 0:
            logger.info("Process Chunks: No chunks input found. Job finished EMPTY.")
            self.status_callback("Status: Finished - No file chunks detected.")
            self.completion_callback(False); return # Signal -> False = No graph change.. DONE early.

         logger.info(f"Processor Starting job: {total_chunks} chunks parallel (Max Workers: {max_workers}). Fn='{original_filename[:30]}'")
         self.status_callback(f"Status: Starting background processing ({total_chunks} chunks)...")

         # Thread Pool execution logic OK.. use it..
         with concurrent.futures.ThreadPoolExecutor( max_workers=max_workers, thread_name_prefix='Lucid_Worker') as executor:
             # Submit ALL tasks - Uses NEW combined Worker Function name OK..
             futures_map = { executor.submit(self._digest_and_embed_chunk_task, chunk, original_filename, i): (i, chunk.get("metadata",{}).get("identifier","?")[:20]) for i, chunk in enumerate(chunks)}

             processed_count = 0; failed_count = 0 # Track Counts good User feedback possible..
             for future in concurrent.futures.as_completed(futures_map):
                   chunk_idx, chunk_disp_id = futures_map[future] # Unpack Context from map OK..
                   try:
                        # Retrieve result (Node or None) from completed WORKER thread... ok.
                        result_node = future.result()
                        if result_node:
                             processed_nodes.append(result_node) # SUCCESS path --> Add to final LIST...
                        else:                             # Task ran BUT returned None = Known Failure Case.--> Logged inside worker! Increment ONLY counter here.. ok..
                            failed_count += 1
                   except Exception as future_exc:
                        # Task ITSELF raised Unhandled Python Exception during execution BAD! --> CRITCAL LOG needed..
                         logger.exception(f"Processor: Background WORKER Task FAILED (Chunk Idx:{chunk_idx} ID:'{chunk_disp_id}') Err: {future_exc}")
                         failed_count += 1 # Failed to complete job chunk level ok..

                   processed_count += 1
                   # --- Progress Status Updates --- Nice to have.. OK keep logic..
                   update_every = max(1, total_chunks // 10) # Update every 10% or each chunk if small number.. ok..
                   if processed_count % update_every == 0 or processed_count == total_chunks:
                       emb_status = 'On' if self.embedding_available else 'Off'
                       stat_msg = f"Status(Bg): Processed {processed_count}/{total_chunks} chunks (Embed={emb_status})..."
                       self.status_callback(stat_msg) # Call callback..

         # --- Processing Loop FINISHED --- >> Post Process Logic --> UPDATE GRAPH! REPORT STATUS!... OK!
         successful_nodes = len(processed_nodes)
         final_log_msg = f"Processor JOB COMPLETE ('{original_filename}' -> {successful_nodes}/{total_chunks} SUCCEEDED nodes processing)."
         if failed_count > 0: final_log_msg += f" **NOTE: {failed_count} chunks encountered ERRORS / skipped**."
         logger.info(final_log_msg)

         # -- Update Memory Graph Object -- Important MUTABLE object ref -> changes are reflected.. OK..
         graph_changed = False # Default assumption No Change.. Ok..
         save_ok=True # Assume SAVE ok -> Set FALSE only if Known SAVE fail.. ok..
         if successful_nodes > 0:
             logger.info(f"Adding {successful_nodes} nodes to the main MemoryGraph instance...")
             graph_changed = True # YES --> Changes WILL Happen..
             nodes_added_count = 0
             for node in processed_nodes:
                 self.memory_graph.add_node(node) # --> MODIFY Shared Graph Object DIRECTLY.. ok..
                 nodes_added_count += 1
             logger.info(f"Added/Updated {nodes_added_count} nodes.")

             # --- >> ATTEMPT TO *SAVE* the UPDATED Graph state NOW! -> Use TRY/CATCH << --- OK.. IMPORTANT STEP.
             save_ok = self._save_graph(); # Call SAVE -> Updates `save_ok` flag based on RETURN value.. YES.. ok..
         else:
             logger.info("No successful new nodes created. MemoryGraph file remains unchanged.")

         # --- Signal FINAL Completion to Controller/External consumer ---
         # FINAL Callback reflects state based on SVE result too... => Complex message Status Good info.. OK..
         final_status_msg = f"Status: Finished '{original_filename}' -> {successful_nodes}/{total_chunks} nodes done."
         if failed_count > 0: final_status_msg+=f" ({failed_count} failures)"
         final_status_msg+= f" | Graph Persist: {'OK' if save_ok else 'FAIL!' if graph_changed else 'No change'}" # Nice status breakdown...
         self.status_callback(final_status_msg) # ---> Trigger Final Status Message..
         self.completion_callback(graph_changed) # ---> Trigger FINAL Callback -> PASS BOOL if CHANGES occurred requires save state.. ok.. DONE.


    def _save_graph(self) -> bool:
         """Saves the current memory graph state to file using MemoryGraph method. Returns Success Bool."""
         logger.debug(f"Processor attempting graph save via memory_graph object -> {MEMORY_GRAPH_PATH}")
         try:
            # DELEGATE save action TO the Graph instance -> contains Save logic now ok.. Better Structure..
            self.memory_graph.save_to_json(MEMORY_GRAPH_PATH)
            logger.info(f"Graph saved OK by Processor -> {MEMORY_GRAPH_PATH}")
            return True # SAVED ok State..
         except Exception as e:
             # Graph SAVE itself FAILED -> Log CRITICAL error.. Important user knows STATE lost ..
            logger.exception(f"CRITICAL Processor FAILURE saving memory graph state to '{MEMORY_GRAPH_PATH}': {e}")
            self.status_callback(f"ALERT STATUS: ERROR saving Memory Graph File!") # <-- Directly Set status maybe useful extra immediate Feedback? ok..
            return False # RETURN Failure State.. Save operation BROKEN..