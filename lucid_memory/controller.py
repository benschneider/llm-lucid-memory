# FILE: lucid_memory/controller.py (Orchestrator - FINAL CLEAN VERSION)

import os
import sys # For checking sys.executable
import json
import subprocess
import threading
import logging
import time # Only needed for minor sleeps
from typing import List, Dict, Any, Optional, Callable, Tuple

# --- Use forward references for type hints to managers if needed ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .managers.config_manager import ConfigManager
    from .managers.component_manager import ComponentManager
    from .managers.server_manager import ServerManager
    from .digestor import Digestor
    from .embedder import Embedder
    from .memory_graph import MemoryGraph

# --- Import MANAGERS from subpackage ---
# Ensure the lucid_memory/managers/__init__.py allows these imports
# or use direct path imports like below. Direct paths are less prone to init issues.
from .managers.config_manager import ConfigManager, API_PRESETS
from .managers.component_manager import ComponentManager
from .managers.server_manager import ServerManager

# --- Import other required types/classes ---
from .memory_graph import MemoryGraph
from .chunker import chunk_file
from .processor import ChunkProcessor
from .memory_node import MemoryNode # Type hint usage

# --- Logging setup ---
# Setup basic logging. Could be configured externally later.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Controller - %(message)s')

# --- Define Constants ---
# Use ConfigManager's path definition now more robustly? Okay keep simple here..
MEMORY_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "memory_graph.json")

class LucidController:
    """
    Orchestrates application flow, manages UI callbacks, holds central state (MemoryGraph),
    and delegates tasks to specialized managers (Config, Components, Server).
    """
    def __init__(self):
        # Initialize Managers first
        self.config_mgr: 'ConfigManager' = ConfigManager()
        self.component_mgr: 'ComponentManager' = ComponentManager(self.config_mgr)
        self.server_mgr: 'ServerManager' = ServerManager(self.config_mgr)

        # Central Data State
        self.memory_graph: 'MemoryGraph' = self._load_memory_graph()

        # Task Management State
        self.processing_active: bool = False
        self.processor_thread: Optional[threading.Thread] = None

        # UI Callbacks / Status
        self.last_status: str = "Initialized."
        self.status_update_callback: Callable[[str], None] = lambda msg: None
        self.graph_update_callback: Optional[Callable[[], None]] = None # Optional callback
        self.completion_callback: Callable[[bool], None] = lambda changed: None

        # Log initial status after all managers load
        initial_status = f"Controller Init OK. Ready = {self.has_required_components}"
        logging.info(initial_status)
        self._set_status(initial_status)


    # --- Internal Helper Methods ----
    def _log_error_and_status(self, error_message: str, status_message: Optional[str] = None, log_level=logging.ERROR) -> None:
        """Helper to log an error and update status."""
        logging.log(log_level, f"Controller: {error_message}")
        status_to_set = status_message or f"Status: Error - {error_message.split('.', 1)[0]}" # Take first part of error msg
        self._set_status(status_to_set)

    def _set_status(self, message: str) -> None:
        """Updates internal status and calls the UI status callback."""
        self.last_status = message
        logging.debug(f"Ctrl Status Set: {message}")
        if callable(self.status_update_callback):
            try:
                self.status_update_callback(message)
            except Exception as ui_cb_err:
                 logging.error(f"ERROR in UI Status Callback execution -> {ui_cb_err}", exc_info=False)

    def get_last_status(self) -> str: return self.last_status


    # --- Public FACADE / Accessor Methods (Delegating to Managers) ---

    ## Configuration Access & Updates ##
    def get_config(self) -> Dict[str, Any]:
        return self.config_mgr.get_config()

    def get_api_presets(self) -> Dict[str, Dict]:
        # Allow UI to access presets directly maybe helpful
        return self.config_mgr.get_api_presets()

    def update_config(self, new_config_values: Dict[str, Any]) -> bool:
        """Updates config via manager; triggers component reload on success."""
        logging.info(f"Controller delegating config update...")
        success = self.config_mgr.update_config(new_config_values)
        if success:
            logging.info("Config update success => Triggering component reload...")
            # ----> CRITICAL: Component reload ensures components use NEW config <----
            self.component_mgr.reload_components()
            self._set_status("Status: Config Saved. Components/Models Refreshed.")
        else:
             self._set_status("Status: Config Update FAILED TO SAVE! Check logs.") # Error already logged by manager
        return success

    ## Component Access & Readiness ##
    def get_available_models(self) -> List[str]:
        return self.component_mgr.get_available_models()

    def refresh_models_list(self) -> None:
         self.component_mgr.reload_components() # Reloading includes fetching models
         model_count = len(self.get_available_models())
         self._set_status(f"Status: Model list refresh attempt complete ({model_count} detected).")

    @property
    def is_digestor_ready(self) -> bool:
        # Delegate readiness check to component manager
        return self.component_mgr.is_digestor_ready

    @property
    def is_embedder_ready(self) -> bool:
         # Delegate readiness check
        return self.component_mgr.is_embedder_ready

    @property
    def has_required_components(self) -> bool:
        # Delegate check - currently based only on digestor
        return self.is_digestor_ready

    ## Server Management ##
    def start_http_server(self) -> bool:
        """Delegates server start to ServerManager and updates status."""
        logging.info("Controller delegating HTTP server start...")
        success = self.server_mgr.start_server()
        if success:
             time.sleep(0.5) # Brief pause might let server start logging/binding
             is_running, status_msg = self.check_http_server_status() # Refresh status
             logging.info(f"Server start initiated. Current poll status: Running={is_running} (Msg='{status_msg}')")
             # Status already updated by check_... method
        else:
             self._set_status("Status: Server Launch FAILED!") # Ensure status covers fail
        return success

    def stop_http_server(self) -> None:
        """Delegates server stop to ServerManager and updates status."""
        logging.info("Controller delegating HTTP server stop...")
        self.server_mgr.stop_server()
        time.sleep(0.2) # Very short pause
        self.check_http_server_status() # Refresh status after stop attempt

    def check_http_server_status(self) -> Tuple[bool, str]:
        """Delegates status check and updates internal status."""
        is_running, status_message = self.server_mgr.check_status()
        self._set_status(f"Status: {status_message}") # Update controller status with result
        return is_running, status_message

    ## Application Cleanup Hook ##
    def cleanup(self):
        logging.info("Controller orchestrator -> Initiating Cleanup Sequence...")
        self.server_mgr.stop_server() # Ensure server is stopped

        proc_thread = self.processor_thread
        if proc_thread and proc_thread.is_alive():
            logging.warning("Cleanup found ACTIVE processor thread, attempting join (5s timeout)...")
            proc_thread.join(timeout=5.0)
            if proc_thread.is_alive():
                 logging.error("Processor thread DID NOT JOIN during cleanup!")
            else:
                 logging.info("Background processor thread joined cleanly.")
        logging.info("Controller Cleanup COMPLETE.")


    # --- Memory Graph Management ---
    # (Owned directly by controller)
    def get_memory_nodes(self) -> Dict[str, 'MemoryNode']:
        # Using forward ref type hint for MemoryNode
        return self.memory_graph.nodes

    def _load_memory_graph(self) -> 'MemoryGraph':
        """Loads graph using MemoryGraph class method."""
        # Using forward ref type hint for MemoryGraph
        graph = MemoryGraph() # Instantiate from imported class
        try:
            graph.load_from_json(MEMORY_GRAPH_PATH) # Load handles backup etc.
        except Exception as e:
             # Log critical error BUT return fresh empty graph for app resilience
             msg = f"CRITICAL: Graph Load FAILED ('{MEMORY_GRAPH_PATH}'). Err: {e}. Starting EMPTY Graph!"
             self._log_error_and_status(msg, status_message="FAIL Graph LOAD!", log_level=logging.CRITICAL)
             graph = MemoryGraph() # Reset to empty
        return graph


    # --- Task / Processing Management ---
    # This logic remains in Controller as it manages the central 'active' flag
    # and coordinates the UI callback structure related to processing start/end.

    def is_processing_active(self) -> bool:
        # **** THIS is the method vs property fix ****
        # Defined WITHOUT @property decorator now.
        """Checks the internal flag indicating if a processing task is running."""
        # Future: Could potentially check self.processor_thread.is_alive() too for more robustness,
        # but relies on thread handle being perfectly managed. Keep simple flag check for now.
        return self.processing_active
        # **** END Fix ****

    def start_processing_pipeline(self, file_path: str) -> None:
        """Validates inputs, then starts background processing task thread."""
        # --- Guard Clauses ---
        if self.is_processing_active(): # Call AS METHOD (no @property)
           self._log_error_and_status("Process already RUNNING. Request Skipped.", log_level=logging.WARNING); return
        if not self.has_required_components: # Use property access here ok
            self._log_error_and_status("REQUIRED Core Component FAIL (Digestor?). Check Config", status_message="Task FAIL-> Component ERROR!"); return

        # --- File Validation ----
        _abs_path = ""; _fname = "";
        try:
            _abs_path=os.path.abspath(file_path)
            if not os.path.isfile(_abs_path): raise FileNotFoundError("Not Valid Path", file_path)
            _fname = os.path.basename(_abs_path)
        except (FileNotFoundError, TypeError, OSError) as f_err:
             self._log_error_and_status(f"FAIL FilePath Validate({f_err}) - '{file_path}'", status_message="ERROR -> BAD FILE Specified"); return

        # ---- LAUNCH Background Thread ---
        self.processing_active = True # Set -> BUSY Flag *BEFORE* launching
        self._set_status(f"Status: Starting Background JOB -> '{_fname}' ...")
        logging.info(f"Spawning Background Processing Thread for file: '{_fname}'")
        try:
            # Ensure Digestor instance EXISTS before passing to thread
            digestor_inst = self.component_mgr.get_digestor()
            if not digestor_inst :
                 raise RuntimeError ("Trying to start Processing - Digestor Instance NOT Available!")
            embedder_inst = self.component_mgr.get_embedder() # Ok if this is None

            thread = threading.Thread(
                target=self._run_background_processing_task,
                # CRITICAL: Pass actual COMPONENT instances & GRAPH ref needed by worker
                args=( _abs_path, _fname, digestor_inst, embedder_inst, self.memory_graph ),
                daemon=True, name=f"PROCESSOR_{_fname[:12]}"
                )
            thread.start()
            self.processor_thread = thread # Store handle
            logging.info(f"--- Background Thread {thread.ident} STARTED OK ---")

        except Exception as launch_e:
             self._log_error_and_status(f"CRITICAL FAIL Background Thread Launch -> ERR: {launch_e}", log_level=logging.CRITICAL, status_message="FATAL Task start ERR!")
             # ---> ROLLBACK STATE carefully <----
             self.processing_active = False; self.processor_thread = None


    def _run_background_processing_task(
             self,
             abs_f_path: str, fname: str,
             # Use forward ref type hints for safety again
             task_digestor: 'Digestor',
             task_embedder: Optional['Embedder'],
             task_memory_graph: 'MemoryGraph'
          ) -> None:
        """INTERNAL WORKER function (target for Thread)."""
        job_finished_ok = False

        try: # -- Main Worker Try Block --
            # --- 1. Read File ---
            self._set_status(f"Status(Bg): Reading '{fname}'...")
            raw_content = ""
            try:
                with open(abs_f_path,'r', encoding='utf-8', errors='ignore') as rf :
                    raw_content = rf.read()
                if not raw_content or raw_content.isspace():
                    logging.warning(f"EMPTY file content '{fname}'. Task Aborted.")
                    self._set_status(f"Status: Finished -> Empty File '{fname}'");
                    job_finished_ok = True; # Mark as finished ok, just no work done
                    if callable(self.completion_callback): self.completion_callback(False);
                    return
            except (IOError, OSError, UnicodeDecodeError) as read_err:
                 self._log_error_and_status(f"File READ Failed '{fname}': {read_err}", status_message=f"FAIL READ: {fname}")
                 if callable(self.completion_callback): self.completion_callback(False); return # signal fail

            # --- 2. Chunk File ---
            self._set_status(f"Status(Bg): Chunking '{fname}'...")
            chunks = []
            try:
                chunks = chunk_file(abs_f_path, raw_content)
                if not chunks :
                    logging.warning(f"No chunks generated '{fname}'. Task Finished.")
                    self._set_status(f"Status: Processed '{fname}' -> 0 Chunks. Done.")
                    job_finished_ok = True; # Mark as finished ok, no work done
                    if callable(self.completion_callback): self.completion_callback(False)
                    return
            except Exception as chunk_err :
                 self._log_error_and_status(f"FATAL Chunking failure occurred '{fname}': {chunk_err}", log_level=logging.CRITICAL, status_message="FATAL Chunking Sys Err!")
                 if callable(self.completion_callback): self.completion_callback(False); return # signal fail

            self._set_status(f"Status(Bg): Chunked OK ({len(chunks)} chunks). Initializing processor...")

            # --- 3. Initialize and Run Processor ---
            # Pre-check (redundant if start guarded, but safe):
            if not task_digestor: raise RuntimeError("Digestor MISSING in worker!")

            processor = ChunkProcessor(
                 digestor=task_digestor, embedder=task_embedder,
                 memory_graph=task_memory_graph,
                 status_callback=self._set_status,
                 completion_callback=self._handle_processor_completion # Link done signal
                )
            processor.process_chunks(chunks, fname) # This BLOCKS and TRIGGERS callback on success
            job_finished_ok = True # If we get here, process_chunks completed OK

        except Exception as worker_err :
              self._log_error_and_status(f"BG Thread RUN FAILED for '{fname}'. ERR->{worker_err}", log_level=logging.CRITICAL, status_message=f"Processing FAILED '{fname}!' Check logs.")
              job_finished_ok = False # Mark failure

        finally: # Essential Cleanup / state reset occurs HERE always
            logging.info(f"---> BG Thread FINISHING '{fname}'. Task Success Signalled Internally? -> {job_finished_ok}")
            # If worker FAILED before processor called success-> *manual* fail completion callback needed
            if not job_finished_ok:
                 logging.error("Worker Error Path -> Manual trigger FAIL completion status.")
                 try:
                      if callable(self.completion_callback):
                          self.completion_callback(graph_changed=False)
                 except Exception as fail_cb_e:
                      logging.exception(f"Error calling FAIL cleanup completion_callback itself! {fail_cb_e}")
            # --> ALWAYS Reset state flags <--
            self.processing_active = False
            self.processor_thread = None
            logging.debug(f"Controller Proc State Flags RESET -> Active=False.")


    # --- COMPLETION Callback (Triggered By Processor Worker) ---
    def _handle_processor_completion(self, graph_changed: bool):
        """Internal callback triggered by Processor on *success*. Relays up."""
        logging.info(f"<-- CTRL Received Proc SUCCESS Signal. Changed Graph Data?={graph_changed}. Relaying...")
        # Relay signal to MAIN callback registed (e.g., by UI)
        if callable(self.completion_callback):
            try:
                self.completion_callback(graph_changed)
            except Exception as main_cb_err:
                logging.exception(f"ERROR in MAIN registered 'completion_callback'! -> {main_cb_err}")
        # NOTE: Flag reset happens in the *finally* block of the worker Thread now, not here


# --- End of LucidController Class ---