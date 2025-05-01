# lucid_memory/controller.py (Confirmed FINAL - Direct State for Progress)

import os
import sys
import json
import subprocess
import threading
import logging
import time
from typing import List, Dict, Any, Optional, Callable, Tuple

# --- Type Hinting & Imports ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .managers.config_manager import ConfigManager
    from .managers.component_manager import ComponentManager
    from .managers.server_manager import ServerManager
    from .digestor import Digestor
    from .embedder import Embedder
    from .memory_graph import MemoryGraph
    from .processor import ChunkProcessor
    from .memory_node import MemoryNode

from .managers.config_manager import ConfigManager, API_PRESETS
from .managers.component_manager import ComponentManager
from .managers.server_manager import ServerManager
from .memory_graph import MemoryGraph
from .chunker import chunk_file
from .processor import ChunkProcessor # Keep import for type hint in worker
from .memory_node import MemoryNode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Controller - %(message)s')
MEMORY_GRAPH_PATH = os.path.join(os.path.dirname(__file__), "memory_graph.json")

class LucidController:
    """
    Orchestrates application flow, manages UI callbacks, holds central state (MemoryGraph),
    and delegates tasks to specialized managers. Maintains internal progress state.
    """
    def __init__(self):
        # Managers
        self.config_mgr: 'ConfigManager' = ConfigManager()
        self.component_mgr: 'ComponentManager' = ComponentManager(self.config_mgr)
        self.server_mgr: 'ServerManager' = ServerManager(self.config_mgr)

        # Central Data State
        self.memory_graph: 'MemoryGraph' = self._load_memory_graph()

        # Task Management State
        self.processing_active: bool = False
        self.processor_thread: Optional[threading.Thread] = None
        # ---> Internal Progress State Attributes <---
        self.current_progress_step: int = 0
        self.total_progress_steps: int = 0
        self.progress_detail_message: str = ""
        # ----------------------------------------------

        # UI Callbacks / Status
        self.last_status: str = "Initializing..."
        # ---> Default lambda accepts **kwargs <---
        self.status_update_callback: Callable[..., None] = lambda **kwargs: None
        # ------------------------------------------
        self.completion_callback: Callable[[bool], None] = lambda changed: None

        initial_status = f"Controller Init OK. Ready = {self.has_required_components}"
        logging.info(initial_status)
        self._set_status(initial_status) # Initial status call is now safe


    # --- Internal Helper Methods ----
    def _log_error_and_status(self, error_message: str, status_message: Optional[str] = None, log_level=logging.ERROR) -> None:
        """Helper to log an error and update the main status message."""
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
        """
        Updates internal state (main status AND progress attributes)
        and calls the main UI status callback (for the general message).
        """
        self.last_status = message
        log_msg = f"Ctrl Status Set: {message}"

        # Update Internal Progress State
        reset_progress = False
        if current_step is not None:
            self.current_progress_step = current_step
            log_msg += f" [Step {current_step}"
            if total_steps is not None:
                 self.total_progress_steps = total_steps
                 log_msg += f"/{total_steps}]"
            else: log_msg += "]"
        elif total_steps is not None:
             self.total_progress_steps = total_steps
             self.current_progress_step = 0
             log_msg += f" [Total Steps Set: {total_steps}]"
        else: reset_progress = True

        if detail is not None:
            self.progress_detail_message = detail
            log_msg += f" Detail: {detail}"
        elif reset_progress:
             self.progress_detail_message = ""

        if reset_progress:
             self.current_progress_step = 0
             self.total_progress_steps = 0
             log_msg += " [Progress Reset]"

        logging.debug(log_msg)

        # Call the MAIN status callback (only for the general message)
        if callable(self.status_update_callback):
            try:
                # Only pass the main message string
                self.status_update_callback(message=message)
            except Exception as ui_cb_err:
                 logging.error(f"ERROR in UI Status Callback function itself: {ui_cb_err}", exc_info=False)
        else:
             logging.warning("No status_update_callback function registered.")

    # --- Public Accessors for Progress State ---
    def get_progress(self) -> Tuple[int, int, str]:
        """Returns the current progress state (step, total, detail)."""
        return (self.current_progress_step, self.total_progress_steps, self.progress_detail_message)

    # --- Public FACADE / Accessor Methods (Delegating to Managers) ---
    def get_config(self) -> Dict[str, Any]: return self.config_mgr.get_config()
    def get_api_presets(self) -> Dict[str, Dict]: return self.config_mgr.get_api_presets()
    def update_config(self, new_config_values: Dict[str, Any]) -> bool:
        success = self.config_mgr.update_config(new_config_values)
        if success:
            logging.info("Config updated via manager, triggering component reload.")
            self.component_mgr.reload_components()
            self._set_status("Status: Config Saved. Components Refreshed.")
        else:
             self._set_status("Status: Config Update FAILED to save!")
        return success

    def get_available_models(self) -> List[str]: return self.component_mgr.get_available_models()
    def refresh_models_list(self) -> None:
         self.component_mgr.reload_components()
         model_count = len(self.get_available_models())
         self._set_status(f"Status: Model list refresh attempt complete ({model_count} found).")

    @property
    def is_digestor_ready(self) -> bool: return self.component_mgr.is_digestor_ready
    @property
    def is_embedder_ready(self) -> bool: return self.component_mgr.is_embedder_ready
    @property
    def has_required_components(self) -> bool: return self.is_digestor_ready

    def start_http_server(self) -> bool:
        success = self.server_mgr.start_server()
        if success: time.sleep(0.5); self.check_http_server_status();
        else: self._set_status ("Status: Server Launch FAILED!")
        return success
    def stop_http_server(self) -> None:
        self.server_mgr.stop_server(); time.sleep(0.2); self.check_http_server_status();
    def check_http_server_status(self) -> Tuple[bool, str]:
         is_running, status_message = self.server_mgr.check_status()
         self._set_status(f"Status: {status_message}")
         return is_running, status_message

    def cleanup(self):
        logging.info("Controller orchestrator -> Initiating Cleanup Sequence...");
        self.server_mgr.stop_server()
        proc_thread = self.processor_thread
        if proc_thread and proc_thread.is_alive():
            logging.warning("Cleanup -> Found ACTIVE processor thread. await JOIN (5s)...")
            proc_thread.join(timeout=5.0)
            if proc_thread.is_alive(): logging.error(">> Processor Thread Join TIMEOUT! unclean exit risk?")
            else: logging.info("Processor thread joined OK.")
        logging.info("Controller Cleanup COMPLETE.")

    def get_memory_nodes(self) -> Dict[str, 'MemoryNode']: return self.memory_graph.nodes
    def _load_memory_graph(self) -> 'MemoryGraph':
        graph = MemoryGraph();
        try: graph.load_from_json(MEMORY_GRAPH_PATH);
        except Exception as e:
             msg = f"CRITICAL: Graph Load FAILED ('{MEMORY_GRAPH_PATH}'). Err:{e} -> Starting EMPTY GRAPH State."
             self._log_error_and_status(msg, status_message="FAIL Graph LOAD!", log_level=logging.CRITICAL)
             graph = MemoryGraph()
        return graph

    def get_last_status(self) -> str: return self.last_status

    # --- Task / Processing Management ---
    def is_processing_active(self) -> bool:
        """Checks the internal flag indicating if a processing task is running."""
        return self.processing_active

    def start_processing_pipeline(self, file_path: str) -> None:
        """Validates inputs, then starts background processing task thread."""
        if self.is_processing_active():
           self._log_error_and_status("Process already RUNNING. Req Skip.", log_level=logging.WARNING); return
        if not self.has_required_components:
            self._log_error_and_status("REQUIRED Core Component FAIL (Digestor?). Check Config", status_message="Task FAIL-> Component ERROR!"); return;

        _abs_path = ""; _fname = "";
        try:
            _abs_path=os.path.abspath(file_path);
            if not os.path.isfile(_abs_path): raise FileNotFoundError("Not Valid Path", file_path);
            _fname = os.path.basename(_abs_path)
        except Exception as f_err:
             self._log_error_and_status(f"FAIL FilePath Validate({f_err}) .. '{file_path}'", status_message="ERROR -> BAD FILE Specified"); return;

        # Reset internal progress state BEFORE starting
        self.current_progress_step = 0
        self.total_progress_steps = 0
        self.progress_detail_message = "Preparing..."
        self.processing_active = True # SET BUSY Flag
        self._set_status(f"Status: Starting JOB -> '{_fname}' ...")
        logging.info(f"Spawning Background Processor Thread for file: '{_fname}'")
        try:
            digestor_inst = self.component_mgr.get_digestor()
            if not digestor_inst : raise RuntimeError ("Digestor Instance NOT Available!")
            embedder_inst = self.component_mgr.get_embedder()

            thread = threading.Thread(
                target=self._run_background_processing_task,
                args=( _abs_path, _fname, digestor_inst, embedder_inst, self.memory_graph ),
                daemon=True, name=f"PROCESSOR_{_fname[:12]}"
                )
            thread.start()
            self.processor_thread = thread
            logging.info(f"--- Background Thread {thread.ident} STARTED OK ---")

        except Exception as launch_e:
             self._log_error_and_status(f"CRITICAL FAIL Background Thread Launch -> ERR: {launch_e}", log_level=logging.CRITICAL, status_message="FATAL Task start ERR!")
             self.processing_active = False; self.processor_thread = None; # ROLLBACK state


    def _run_background_processing_task(
             self,
             abs_f_path: str, fname: str,
             task_digestor: 'Digestor',
             task_embedder: Optional['Embedder'],
             task_memory_graph: 'MemoryGraph'
          ) -> None:
        """INTERNAL WORKER function (target for Thread)."""
        job_finished_ok = False
        from .processor import ChunkProcessor # Local import ok

        try: # -- Main Worker Try Block --
            # --- 1. Read File ---
            self._set_status(f"Status(Bg): Reading '{fname}'...")
            raw_content = ""
            try:
                with open(abs_f_path,'r', encoding='utf-8', errors='ignore') as rf : raw_content = rf.read()
                if not raw_content or raw_content.isspace():
                    logging.warning(f"EMPTY file content '{fname}'. Task Aborted.")
                    self._set_status(f"Status: Finished -> Empty File '{fname}'", detail="Empty file.")
                    job_finished_ok = True;
                    if callable(self.completion_callback): self.completion_callback(False);
                    return
            except (IOError, OSError, UnicodeDecodeError) as read_err:
                 self._log_error_and_status(f"File READ Failed '{fname}': {read_err}", status_message=f"FAIL READ: {fname}")
                 if callable(self.completion_callback): self.completion_callback(False); return

            # --- 2. Chunk File ---
            self._set_status(f"Status(Bg): Chunking '{fname}'...", detail="Analyzing structure...")
            chunks = []
            try:
                chunks = chunk_file(abs_f_path, raw_content)
                if not chunks :
                    logging.warning(f"No chunks generated '{fname}'. Task Finished.")
                    self._set_status(f"Status: Processed '{fname}' -> 0 Chunks. Done.", detail="No processable chunks found.")
                    job_finished_ok = True;
                    if callable(self.completion_callback): self.completion_callback(False)
                    return
            except Exception as chunk_err :
                 self._log_error_and_status(f"FATAL Chunking failure '{fname}': {chunk_err}", log_level=logging.CRITICAL, status_message="FATAL Chunking Sys Err!")
                 if callable(self.completion_callback): self.completion_callback(False); return

            self._set_status(f"Status(Bg): Chunked OK ({len(chunks)} found). Initializing processor...", detail=f"{len(chunks)} chunks found.")

            # --- 3. Initialize and Run Processor ---
            if not task_digestor: raise RuntimeError("Digestor MISSING in worker!")

            processor = ChunkProcessor(
                 digestor=task_digestor, embedder=task_embedder,
                 memory_graph=task_memory_graph,
                 status_callback=self._set_status, # Pass controller's method
                 completion_callback=self._handle_processor_completion
                )
            processor.process_chunks(chunks, fname) # BLOCKS and calls status_callback
            job_finished_ok = True # Assume success if process_chunks returns

        except Exception as worker_err :
              self._log_error_and_status(f"BG Thread RUN CRITICAL FAILED '{fname}'. ERR->{worker_err}", log_level=logging.CRITICAL, status_message=f"Processing FAILED '{fname}!' Check logs.")
              job_finished_ok = False

        finally:
            logging.info(f"---> BG Thread FINISHING '{fname}'. Task Success Signalled Internally? -> {job_finished_ok}")
            if not job_finished_ok:
                 logging.error("Worker FAILURE/ERROR Path -> Manual trigger FAIL completion status.")
                 try:
                      if callable(self.completion_callback): self.completion_callback(graph_changed=False)
                 except Exception as final_cb_err: logging.exception(f"Error calling FAIL cleanup completion_callback! {final_cb_err}")
            # Reset progress state on exit regardless of success/fail
            self.current_progress_step = 0
            self.total_progress_steps = 0
            self.progress_detail_message = ""
            # Reset main flags
            self.processing_active = False
            self.processor_thread = None
            logging.debug(f"Controller state flags/progress reset post-processing.")


    # --- COMPLETION Callback (Triggered By Processor Worker) ---
    def _handle_processor_completion(self, graph_changed: bool):
        """Internal callback triggered by Processor on success. Relays up."""
        logging.info(f"<-- CTRL Received Proc SUCCESS Signal. Changed Graph Data?={graph_changed}.--> Relay MAIN CB...")
        # Reset progress state HERE as well, ensuring it's cleared on SUCCESS path
        self.current_progress_step = 0
        self.total_progress_steps = 0
        self.progress_detail_message = ""
        # Relay signal to MAIN callback registered (e.g., by UI)
        if callable(self.completion_callback):
            try: self.completion_callback(graph_changed);
            except Exception as main_cb_err: logging.exception(f"ERROR in MAIN registered 'completion_callback'! -> {main_cb_err}")

# --- End of LucidController Class ---