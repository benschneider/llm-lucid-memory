import os
import sys
import json
import subprocess
import threading
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple, TYPE_CHECKING
from .digestor import Digestor
from .embedder import Embedder
from .managers.config_manager import ConfigManager
from .managers.component_manager import ComponentManager
from .memory_graph import MemoryGraph
from .chunker import chunk_file, chunk as lib_chunk_content # Renamed to avoid conflict
from .processor import ChunkProcessor
from .memory_node import MemoryNode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Controller - %(message)s')

# Default path will be inside the lucid_memory package.
LUCID_MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEMORY_GRAPH_PATH = os.path.join(LUCID_MEMORY_DIR, "memory_graph.json")

# Expose functions for API (these were stubs before, now they'll call controller methods)
# These are top-level functions that would instantiate a controller if needed,
# or the API will use a single controller instance.
# For api.py, it's better if it uses an instance of LucidController.
# So, these specific top-level functions might not be strictly needed if api.py directly uses LucidController methods.
# Let's remove the __all__ for now and have api.py use LucidController instance.

class LucidController:
    """
    Orchestrates application flow, manages UI callbacks, holds central state (MemoryGraph),
    and delegates tasks to specialized managers. Maintains internal progress state.
    """
    def __init__(self, memory_graph_path: str = DEFAULT_MEMORY_GRAPH_PATH):
        # Managers
        self.config_mgr: 'ConfigManager' = ConfigManager()
        self.component_mgr: 'ComponentManager' = ComponentManager(self.config_mgr)
        self.server_mgr: 'ServerManager' = ServerManager(self.config_mgr)

        # Central Data State
        self.memory_graph_path = memory_graph_path
        self.memory_graph: 'MemoryGraph' = self._load_memory_graph()

        # Task Management State
        self.processing_active: bool = False
        self.processor_thread: Optional[threading.Thread] = None
        self.current_progress_step: int = 0
        self.total_progress_steps: int = 0
        self.progress_detail_message: str = ""

        self.last_status: str = "Controller Initializing..."
        self.status_update_callback: Callable[..., None] = lambda **kwargs: None # For Streamlit UI
        self.completion_callback: Callable[[bool], None] = lambda changed: None # For Streamlit UI

        initial_status = f"Controller Init OK. Digestor: {self.is_digestor_ready}, Embedder: {self.is_embedder_ready}"
        logging.info(initial_status)
        self._set_status(initial_status)

    # --- Internal Helper Methods ----
    def _log_error_and_status(self, error_message: str, status_message: Optional[str] = None, log_level=logging.ERROR) -> None:
        logging.log(log_level, f"Controller: {error_message}")
        status_to_set = status_message or f"Status: Error - {error_message.split('.', 1)[0]}"
        self._set_status(status_to_set)

    def _set_status(
            self,
            message: str,
            current_step: Optional[int] = None,
            total_steps: Optional[int] = None,
            detail: Optional[str] = None
        ) -> None:
        self.last_status = message
        log_msg = f"Ctrl Status Set: {message}"

        reset_progress = False
        if current_step is not None: self.current_progress_step = current_step
        if total_steps is not None: self.total_progress_steps = total_steps
        if detail is not None: self.progress_detail_message = detail
        
        if current_step is None and total_steps is None and detail is None: # Simple message update
            reset_progress = True

        if reset_progress:
             self.current_progress_step = 0
             self.total_progress_steps = 0
             self.progress_detail_message = ""
             log_msg += " [Progress Reset]"
        else:
            log_msg += f" [Step {self.current_progress_step or 0}/{self.total_progress_steps or 0}] Detail: {self.progress_detail_message or 'N/A'}"


        logging.debug(log_msg)
        if callable(self.status_update_callback):
            try:
                self.status_update_callback(message=message) # Keep simple for basic UI
            except Exception as ui_cb_err:
                 logging.error(f"ERROR in UI Status Callback function itself: {ui_cb_err}", exc_info=False)

    # --- Public Accessors for Progress State ---
    def get_progress(self) -> Tuple[int, int, str]:
        return (self.current_progress_step, self.total_progress_steps, self.progress_detail_message)

    # --- Public FACADE / Accessor Methods (Delegating to Managers) ---
    def get_config(self) -> Dict[str, Any]: return self.config_mgr.get_config()
    def get_api_presets(self) -> Dict[str, Dict]: return self.config_mgr.get_api_presets()
    
    def update_config_values(self, new_config_values: Dict[str, Any]) -> bool: # Renamed for clarity
        success = self.config_mgr.update_config(new_config_values)
        if success:
            logging.info("Config updated via manager, triggering component reload.")
            self.component_mgr.reload_components() # Reload components like Digestor, Embedder
            # Also need to re-evaluate server settings if port changed, etc.
            self.server_mgr.config_mgr = self.config_mgr # Ensure server manager has updated config_mgr if it makes a copy
            self._set_status(f"Status: Config Saved. Components Refreshed. Digestor: {self.is_digestor_ready}, Embedder: {self.is_embedder_ready}")
        else:
             self._set_status("Status: Config Update FAILED to save!")
        return success

    def get_available_models(self) -> List[str]: return self.component_mgr.get_available_models()
    def refresh_models_list(self) -> None:
         self.component_mgr.reload_components() # This already calls _fetch_models_from_backend
         model_count = len(self.get_available_models())
         self._set_status(f"Status: Model list refresh attempt complete ({model_count} found).")

    @property
    def is_digestor_ready(self) -> bool: return self.component_mgr.is_digestor_ready
    @property
    def is_embedder_ready(self) -> bool: return self.component_mgr.is_embedder_ready
    @property
    def has_required_components(self) -> bool: return self.is_digestor_ready # Embedder is optional

    def start_http_server(self) -> bool:
        success = self.server_mgr.start_server()
        if success: time.sleep(0.5); self.check_http_server_status();
        else: self._set_status ("Status: HTTP Proxy Server Launch FAILED!")
        return success
    def stop_http_server(self) -> None:
        self.server_mgr.stop_server(); time.sleep(0.2); self.check_http_server_status();
    def check_http_server_status(self) -> Tuple[bool, str]:
         is_running, status_message = self.server_mgr.check_status()
         self._set_status(f"Status: Proxy Server - {status_message}")
         return is_running, status_message

    def cleanup(self):
        logging.info("Controller: Initiating Cleanup Sequence...")
        self.stop_http_server() # Ensure server is stopped
        
        if self.processor_thread and self.processor_thread.is_alive():
            logging.warning("Cleanup: Active processor thread found. Attempting to join (5s timeout)...")
            self.processor_thread.join(timeout=5.0)
            if self.processor_thread.is_alive():
                logging.error(">> Processor Thread Join TIMEOUT! Unclean exit for processing task possible.")
            else:
                logging.info("Processor thread joined successfully.")
        self.processor_thread = None
        self.processing_active = False
        logging.info("Controller Cleanup COMPLETE.")

    def get_all_memory_nodes(self) -> Dict[str, 'MemoryNode']: # Renamed for clarity
        return self.memory_graph.nodes

    def _load_memory_graph(self) -> 'MemoryGraph':
        graph = MemoryGraph()
        try:
            graph.load_from_json(self.memory_graph_path)
            self._set_status(f"Memory graph loaded from {self.memory_graph_path} ({len(graph.nodes)} nodes).")
        except Exception as e:
             msg = f"Controller: Graph Load FAILED ('{self.memory_graph_path}'). Err:{e} -> Starting EMPTY GRAPH State."
             self._log_error_and_status(msg, status_message="FAIL Graph LOAD!", log_level=logging.WARNING) # Not CRITICAL to stop controller init
             graph = MemoryGraph() # Ensure it's a new empty graph
        return graph

    def get_last_status(self) -> str: return self.last_status

    # --- Task / Processing Management ---
    def is_processing_active(self) -> bool:
        return self.processing_active

    def start_processing_pipeline_async(self, file_path: str) -> None: # Renamed for clarity
        """Validates inputs, then starts ASYNCHRONOUS background processing task thread. (For UI)"""
        if self.is_processing_active():
           self._log_error_and_status("Processing task already RUNNING. Request skipped.", log_level=logging.WARNING); return
        if not self.has_required_components:
            self._log_error_and_status("REQUIRED Core Component (Digestor) not ready. Check Config.", status_message="Task FAIL -> Component ERROR!"); return;

        _abs_path = ""; _fname = "";
        try:
            _abs_path=os.path.abspath(file_path);
            if not os.path.isfile(_abs_path): raise FileNotFoundError(f"Not a Valid File Path: {file_path}");
            _fname = os.path.basename(_abs_path)
        except Exception as f_err:
             self._log_error_and_status(f"FAIL FilePath Validate({f_err}) for '{file_path}'", status_message="ERROR -> BAD FILE Specified"); return;

        self.processing_active = True
        self._set_status(f"Status: Starting ASYNC Processing JOB for '{_fname}' ...", 0, 1, "Preparing...") # Initial progress
        logging.info(f"Spawning Background Processor Thread for file: '{_fname}'")
        try:
            digestor_inst = self.component_mgr.get_digestor()
            if not digestor_inst : raise RuntimeError ("Digestor Instance NOT Available for async processing!")
            embedder_inst = self.component_mgr.get_embedder() # Can be None

            thread = threading.Thread(
                target=self._run_background_processing_task,
                args=( _abs_path, _fname, digestor_inst, embedder_inst, self.memory_graph ),
                daemon=True, name=f"PROCESSOR_ASYNC_{_fname[:12]}"
                )
            thread.start()
            self.processor_thread = thread
            logging.info(f"--- Background Thread {thread.name} (ID: {thread.ident}) STARTED OK ---")

        except Exception as launch_e:
             self._log_error_and_status(f"CRITICAL FAIL Background Thread Launch -> ERR: {launch_e}", log_level=logging.CRITICAL, status_message="FATAL Task start ERR!")
             self.processing_active = False; self.processor_thread = None;

    def _run_background_processing_task(
             self,
             abs_f_path: str, fname: str,
             task_digestor: 'Digestor',
             task_embedder: Optional['Embedder'],
             task_memory_graph: 'MemoryGraph'
          ) -> None:
        job_finished_ok = False
        # from .processor import ChunkProcessor # Already imported at top level

        try:
            self._set_status(f"Status(Bg): Reading '{fname}'...", detail="Reading file content...")
            raw_content = ""
            try:
                with open(abs_f_path,'r', encoding='utf-8', errors='ignore') as rf : raw_content = rf.read()
                if not raw_content or raw_content.isspace():
                    logging.warning(f"EMPTY file content '{fname}'. Task Aborted.")
                    self._set_status(f"Status: Finished -> Empty File '{fname}'", detail="Empty file.")
                    job_finished_ok = True; # No change to graph
                    if callable(self.completion_callback): self.completion_callback(False);
                    return
            except Exception as read_err: # Catch any read error
                 self._log_error_and_status(f"File READ Failed for '{fname}': {read_err}", status_message=f"FAIL READ: {fname}")
                 if callable(self.completion_callback): self.completion_callback(False); return

            self._set_status(f"Status(Bg): Chunking '{fname}'...", detail="Analyzing structure...")
            chunks = []
            try:
                chunks = chunk_file(abs_f_path, raw_content) # chunk_file handles file type detection
                if not chunks :
                    logging.warning(f"No chunks generated for '{fname}'. Task Finished.")
                    self._set_status(f"Status: Processed '{fname}' -> 0 Chunks. Done.", detail="No processable chunks found.")
                    job_finished_ok = True;
                    if callable(self.completion_callback): self.completion_callback(False)
                    return
            except Exception as chunk_err :
                 self._log_error_and_status(f"FATAL Chunking failure for '{fname}': {chunk_err}", log_level=logging.CRITICAL, status_message="FATAL Chunking Sys Err!")
                 if callable(self.completion_callback): self.completion_callback(False); return

            self._set_status(f"Status(Bg): Chunked OK ({len(chunks)} found). Initializing processor...", detail=f"{len(chunks)} chunks found.")

            if not task_digestor: raise RuntimeError("Digestor MISSING in background worker!")

            processor = ChunkProcessor(
                 digestor=task_digestor, embedder=task_embedder,
                 memory_graph=task_memory_graph,
                 status_callback=self._set_status, # Pass controller's method for UI updates
                 completion_callback=self._handle_processor_completion, # Controller's internal handler
                 memory_graph_path_override=self.memory_graph_path # Pass the correct graph path
                )
            processor.process_chunks(chunks, fname) # BLOCKS and calls status_callback
            job_finished_ok = True 

        except Exception as worker_err :
              self._log_error_and_status(f"BG Thread RUN CRITICAL FAILED for '{fname}'. ERR->{worker_err}", log_level=logging.CRITICAL, status_message=f"Processing FAILED for '{fname}!' Check logs.")
              job_finished_ok = False

        finally:
            logging.info(f"---> BG Thread ({threading.current_thread().name}) FINISHING for '{fname}'. Task Success Signalled Internally? -> {job_finished_ok}")
            if not job_finished_ok:
                 logging.error("Worker FAILURE/ERROR Path -> Manual trigger FAIL completion status.")
                 try:
                      if callable(self.completion_callback): self.completion_callback(graph_changed=False)
                 except Exception as final_cb_err: logging.exception(f"Error calling FAIL cleanup completion_callback! {final_cb_err}")
            
            self._set_status(
                f"Status: Async processing for '{fname}' {'completed' if job_finished_ok else 'FAILED'}.",
                detail="Finished." if job_finished_ok else "Error encountered."
            ) # Reset progress by not passing step/total

            self.processing_active = False # Crucial: reset flag
            self.processor_thread = None # Crucial: clear thread reference
            logging.debug(f"Controller state flags/progress reset post-async-processing for '{fname}'.")

    def _handle_processor_completion(self, graph_changed: bool):
        """Internal callback triggered by Processor on success. Relays up to UI's callback."""
        logging.info(f"<-- CTRL Received Processor SUCCESS Signal. Graph Changed?={graph_changed}. --> Relaying to MAIN (UI) Completion CB...")
        self._set_status( # Final status update for UI
            f"Status: Processing finished. Graph data {'updated' if graph_changed else 'unchanged'}.",
            detail="Successfully completed all chunks."
        )
        if callable(self.completion_callback):
            try: self.completion_callback(graph_changed);
            except Exception as main_cb_err: logging.exception(f"ERROR in MAIN registered 'completion_callback'! -> {main_cb_err}")

    # --- Synchronous methods for API ---
    def chunk_content_synchronously(self, source_code: str, file_identifier: str = "api_input") -> List[Dict[str, Any]]:
        """Chunks source code string directly (for API /chunk)."""
        # Uses the chunker's `chunk` function designed for string input.
        # The lib_chunk_content is an alias for lucid_memory.chunker.chunk
        logging.info(f"Controller: Synchronously chunking content for identifier: {file_identifier}")
        return lib_chunk_content(source_code, file_identifier=file_identifier)


    def process_chunks_synchronously(self, chunks_data: List[Dict[str, Any]], original_filename: str) -> List[MemoryNode]:
        """
        Processes chunks synchronously (for API /process).
        This will block until completion and return the list of MemoryNode objects.
        """
        if self.is_processing_active():
            raise Exception("Another processing task is already active. Please wait.")
        if not self.has_required_components:
            raise Exception("Digestor (required component) is not ready. Check configuration.")

        self.processing_active = True
        logging.info(f"Controller: Starting SYNCHRONOUS processing for '{original_filename}' ({len(chunks_data)} chunks).")
        
        processed_nodes_list: List[MemoryNode] = []
        graph_changed_by_sync_process = False

        # Define simple internal callbacks for synchronous processing
        def sync_status_cb(message: str, current_step: Optional[int]=None, total_steps: Optional[int]=None, detail: Optional[str]=None):
            log_msg = f"SYNC_PROC_STATUS: {message}"
            if current_step is not None and total_steps is not None: log_msg += f" ({current_step}/{total_steps})"
            if detail: log_msg += f" - {detail}"
            logging.info(log_msg) # Just log for sync mode

        def sync_completion_cb(graph_changed: bool):
            nonlocal graph_changed_by_sync_process # Modify outer scope variable
            graph_changed_by_sync_process = graph_changed
            logging.info(f"SYNC_PROC_COMPLETED: Graph changed: {graph_changed}")

        try:
            digestor_inst = self.component_mgr.get_digestor()
            if not digestor_inst: raise RuntimeError("Digestor instance not available for synchronous processing!")
            embedder_inst = self.component_mgr.get_embedder() # Can be None

            # Create a temporary MemoryGraph instance or use the main one.
            # Using the main one means changes are immediately reflected and saved.
            # If API calls should be isolated, a temporary graph could be used.
            # For now, use the main graph.
            
            processor = ChunkProcessor(
                 digestor=digestor_inst, embedder=embedder_inst,
                 memory_graph=self.memory_graph, # Use controller's main graph
                 status_callback=sync_status_cb,
                 completion_callback=sync_completion_cb,
                 memory_graph_path_override=self.memory_graph_path # Ensure it saves to the right place
                )
            
            # The ChunkProcessor's process_chunks method returns None.
            # It updates the graph internally and calls completion_callback.
            # We need to collect the nodes that were added/updated.
            # One way is to get all nodes from graph before and after, then diff.
            # Or, modify ChunkProcessor to return the list of processed nodes.
            # For simplicity, let's assume ChunkProcessor saves to self.memory_graph.
            # We can then iterate through the input `chunks_data` and try to find corresponding nodes.
            # A cleaner way: modify ChunkProcessor to return successfully processed nodes.
            # Let's go with modifying ChunkProcessor later. For now, a workaround.
            
            # Workaround: To get the nodes created by this specific call, we could
            # snapshot node IDs before, then process, then see what's new.
            # This is a bit fragile if IDs are not perfectly predictable.
            # For now, process_chunks will modify self.memory_graph directly.
            # The API will then ask for all nodes or nodes matching some criteria.
            
            # For this specific method, let's assume `process_chunks` now returns the nodes it made.
            # This requires a change in ChunkProcessor.
            # If ChunkProcessor CANNOT be changed to return nodes:
            # We would call process_chunks, then maybe try to fetch nodes from self.memory_graph
            # based on expected IDs from chunks_data, which is complex.
            # Let's assume for now that process_chunks in the sync context will give us the nodes.
            
            # *** TEMPORARY: ChunkProcessor.process_chunks does not return nodes.
            # *** It updates the graph and calls callbacks.
            # *** For the API, we want the actual nodes.
            # *** This part needs ChunkProcessor to be adapted or a different approach.

            # Let's simulate what ChunkProcessor would do and return the nodes.
            # This is a more direct implementation for the synchronous API.
            
            max_workers = max(1, min(4, (os.cpu_count() or 1))) # Fewer workers for sync API call?
            temp_processed_nodes: List[MemoryNode] = []

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='Lucid_Sync_Worker') as executor:
                futures = []
                for i, chunk_item in enumerate(chunks_data):
                    futures.append(executor.submit(processor._digest_and_embed_chunk_task, chunk_item, original_filename, i))
                
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    try:
                        node = future.result()
                        if node:
                            temp_processed_nodes.append(node)
                            self.memory_graph.add_node(node) # Add to main graph
                            logging.info(f"Sync processed and added node: {node.id}")
                        else:
                            logging.warning(f"Sync processing: chunk {i} resulted in no node.")
                    except Exception as e:
                        logging.error(f"Error processing chunk {i} synchronously: {e}", exc_info=True)
            
            if temp_processed_nodes:
                processor._save_graph() # Save graph if nodes were added/changed
                graph_changed_by_sync_process = True
            
            processed_nodes_list = temp_processed_nodes


        except Exception as e:
            logging.error(f"Error during synchronous processing for '{original_filename}': {e}", exc_info=True)
            self.processing_active = False # Ensure flag is reset on error
            raise # Re-throw exception to be caught by API layer
        finally:
            self.processing_active = False # Reset flag
            logging.info(f"Controller: Finished SYNCHRONOUS processing for '{original_filename}'. Nodes processed: {len(processed_nodes_list)}. Graph changed: {graph_changed_by_sync_process}")

        return processed_nodes_list

    def add_memory_node_data(self, memory_node_data: Dict[str, Any]) -> bool:
        """
        Creates a MemoryNode from dictionary data and adds it to the graph (for API /add_memory_node).
        Saves the graph after adding.
        """
        try:
            # Add default ID if missing, though API client should provide it.
            if 'id' not in memory_node_data :
                 memory_node_data['id'] = f"api_added_node_{int(time.time())}" # Generate an ID
            
            node = MemoryNode.from_dict(memory_node_data)
            self.memory_graph.add_node(node)
            self.memory_graph.save_to_json(self.memory_graph_path) # Save after adding
            logging.info(f"Controller: Added memory node '{node.id}' via API and saved graph.")
            return True
        except Exception as e:
            logging.error(f"Controller: Error adding memory node via API: {e}", exc_info=True)
            return False
