# lucid_memory/controller.py

import os
import json
import subprocess
import threading
import logging
from typing import List, Dict, Any, Optional, Callable

# Import components used by the controller
from .memory_graph import MemoryGraph
from .chunker import chunk_file
from .digestor import Digestor
from .processor import ChunkProcessor
from .memory_node import MemoryNode # Needed for type hinting

# --- Configuration ---
# Share constants/defaults or load config independently here as well
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "proxy_config.json")
MEMORY_GRAPH_PATH = "memory_graph.json"
DEFAULT_CONFIG = {
    "backend_url": "http://localhost:11434/v1/chat/completions",
    "model_name": "mistral",
    "local_proxy_port": 8000
}

class LucidController:
    """
    Manages the application state, orchestrates background tasks (digestion, server),
    and interacts with core Lucid Memory components.
    Designed to be used by a UI layer (like the Tkinter GUI).
    """
    def __init__(self):
        self.config: Dict[str, Any] = self._load_config()
        self.memory_graph: MemoryGraph = self._load_memory_graph()
        self.digestor_available: bool = self._check_digestor_availability()
        self.server_process: Optional[subprocess.Popen] = None
        self.processing_active: bool = False
        self.last_status: str = "Initialized"

        # Callbacks to be set by the UI layer
        self.status_update_callback: Callable[[str], None] = lambda msg: None # No-op default
        self.graph_update_callback: Callable[[], None] = lambda: None          # No-op default

    # --- Configuration Management ---

    def get_config(self) -> Dict[str, Any]:
        """Returns the current configuration."""
        return self.config.copy()

    def _load_config(self) -> Dict[str, Any]:
        """Loads config from file or returns default (Internal)."""
        cfg = DEFAULT_CONFIG.copy()
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f: loaded_cfg = json.load(f)
                cfg.update(loaded_cfg) # Merge loaded config over defaults
                logging.info(f"Controller: Loaded config from {CONFIG_PATH}")
            except Exception as e: logging.error(f"Controller: Error parsing {CONFIG_PATH}: {e}. Using defaults.", exc_info=False)
        else: logging.warning(f"Controller: Config {CONFIG_PATH} not found. Using/creating default.")
        # Attempt to save default if not already present
        if not os.path.exists(CONFIG_PATH): self._save_config(cfg)
        return cfg

    def _save_config(self, config_to_save: Dict[str, Any]) -> bool:
        """Saves the given config to file (Internal). Returns True on success."""
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config_to_save, f, indent=2)
            logging.info(f"Controller: Saved configuration to {CONFIG_PATH}")
            return True
        except Exception as e:
            logging.error(f"Controller: Failed to save configuration: {e}", exc_info=True)
            self._set_status(f"Status: Error saving config!")
            return False

    def update_config(self, new_config_values: Dict[str, Any]) -> bool:
        """Updates specific config values and saves the full config."""
        # Validate basic required keys exist
        if not new_config_values.get('backend_url') or not new_config_values.get('model_name'):
            logging.error("Controller: Update failed - missing URL or Model Name.")
            return False
        try: # Validate port
            int(new_config_values.get('local_proxy_port', 8000))
        except ValueError:
            logging.error("Controller: Update failed - invalid port.")
            return False

        self.config.update(new_config_values) # Merge updates
        if self._save_config(self.config):
            self.digestor_available = self._check_digestor_availability() # Re-check readiness
            status = "Status: Config saved. Digestor " + ("ready." if self.digestor_available else "NOT ready (Check config/logs).")
            self._set_status(status)
            return True
        return False

    # --- Digestor Readiness ---

    def is_digestor_ready(self) -> bool:
        """Returns whether the digestor seems ready based on current config."""
        return self.digestor_available

    def _check_digestor_availability(self) -> bool:
        """Checks prerequisites for Digestor initialization."""
        if not self.config.get('backend_url') or not self.config.get('model_name'):
            logging.warning("Controller: Digestor unavailable due to missing config.")
            return False
        try:
            # Try creating a temporary Digestor instance to check for loading errors (e.g., prompts file)
            Digestor(self.config) # Assume Digestor accepts config now
            logging.info("Controller: Digestor availability check passed.")
            return True
        except Exception as e:
            logging.error(f"Controller: Digestor availability check failed: {e}", exc_info=True)
            return False

    # --- Memory Graph Management ---

    def get_memory_nodes(self) -> Dict[str, MemoryNode]:
         """Returns the currently loaded memory nodes."""
         # Return a copy if mutation by UI is a concern, but likely okay for display
         return self.memory_graph.nodes

    def _load_memory_graph(self) -> MemoryGraph:
        """Loads memory graph from file (Internal)."""
        graph = MemoryGraph()
        if not os.path.exists(MEMORY_GRAPH_PATH):
             logging.info(f"Controller: Memory file {MEMORY_GRAPH_PATH} not found.")
             return graph
        try:
            graph.load_from_json(MEMORY_GRAPH_PATH)
            logging.info(f"Controller: Loaded {len(graph.nodes)} nodes from {MEMORY_GRAPH_PATH}")
        except Exception as e: # Catch load/parse errors
            logging.warning(f"Controller: Error loading {MEMORY_GRAPH_PATH}: {e}. Starting empty.", exc_info=False)
        return graph

    # Note: Saving is handled by the processor completion callback now

    # --- Status Management ---

    def _set_status(self, message: str):
        """Updates internal status and calls the UI callback."""
        self.last_status = message
        if self.status_update_callback:
            try:
                self.status_update_callback(message)
            except Exception as e:
                 logging.error(f"Controller: Error executing status callback: {e}", exc_info=False)

    def get_last_status(self) -> str:
        return self.last_status

    def is_processing(self) -> bool:
         """Checks if a digestion process is currently active."""
         return self.processing_active

    # --- Digestion Orchestration ---

    def start_digestion_for_file(self, file_path: str):
        """Starts the chunking and parallel digestion process for a file."""
        if not self.digestor_available:
            logging.error("Controller: Cannot start digestion, Digestor not ready.")
            self._set_status("Status: Digestor Error - Check Config.")
            # Consider raising an error or returning status?
            return
        if self.processing_active:
            logging.warning("Controller: Digestion already in progress.")
            self._set_status("Status: Busy - Already processing.")
            return

        self.processing_active = True
        self._set_status(f"Status: Starting processing for {os.path.basename(file_path)}...")

        # Run the actual processing in a background thread
        thread = threading.Thread(
            target=self._run_processor_task,
            args=(file_path,),
            daemon=True
        )
        thread.start()

    def _run_processor_task(self, file_path: str):
        """The function executed in the background thread for processing."""
        current_digestor = None # Initialize
        try:
             # 1. Read File
             self._set_status(f"Status: Reading {os.path.basename(file_path)}...")
             with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                  raw_text = f.read()

             # 2. Chunk File
             self._set_status(f"Status: Chunking {os.path.basename(file_path)}...")
             chunks = chunk_file(file_path, raw_text)
             if not chunks:
                  logging.warning(f"Controller: File yielded no chunks: {file_path}")
                  self._set_status("Status: Skipped - File yielded no chunks.")
                  self.processing_active = False
                  # Ensure completion callback is called even if no work done
                  if self.completion_callback: self.completion_callback(False)
                  return

             self._set_status(f"Status: Chunked ({len(chunks)}). Initializing processor...")

             # 3. Initialize Components for Processor
             # Re-create Digestor to potentially pick up config changes
             try:
                  current_digestor = Digestor(self.config) # Pass config
             except Exception as e:
                  logging.error(f"Controller: Failed to create Digestor for processor: {e}", exc_info=True)
                  self._set_status("Status: Error - Failed Digestor init")
                  self.processing_active = False
                  if self.completion_callback: self.completion_callback(False)
                  return

             processor = ChunkProcessor(
                 digestor=current_digestor,
                 memory_graph=self.memory_graph, # Use the controller's graph instance
                 status_callback=self._set_status, # Use internal status update method
                 completion_callback=self._handle_processor_completion # Use internal handler
             )

             # 4. Run the Processor (This blocks the *worker* thread)
             processor.process_chunks(chunks, os.path.basename(file_path))

        except FileNotFoundError:
             logging.error(f"Controller: File not found for processing: {file_path}")
             self._set_status("Status: Error - File not found.")
             self.processing_active = False
             if self.completion_callback: self.completion_callback(False)
        except Exception as e:
             logging.error(f"Controller: Error during file processing task for {file_path}: {e}", exc_info=True)
             self._set_status("Status: Error during processing. Check Logs.")
             self.processing_active = False
             if self.completion_callback: self.completion_callback(False) # Signal completion even on error
        # Note: `finally` block isn't strictly needed here because completion is signaled
        # by theChunkProcessor's completion callback calling _handle_processor_completion

    def _handle_processor_completion(self, graph_changed: bool):
        """Internal handler for when the ChunkProcessor finishes."""
        logging.info(f"Controller: Handling processor completion. Graph changed: {graph_changed}")
        self.processing_active = False # Mark processing as finished
        # We already updated the status message inside process_chunks via callback
        # self._set_status("Status: Processing Complete.") # Can override processor final status if needed

        # Call the UI's completion callback (which might trigger a UI refresh)
        if self.completion_callback:
            try:
                 self.completion_callback(graph_changed)
            except Exception as e:
                 logging.error(f"Controller: Error executing completion callback: {e}", exc_info=False)


    # --- Server Management ---

    def start_server(self) -> bool:
        """"Starts the Uvicorn proxy server. Returns True if potentially started."""
        if self.server_process and self.server_process.poll() is None:
            logging.warning("Controller: Start server called, but already running.")
            self._set_status("Status: Server already running.")
            return True # It's already started

        py_exe = os.sys.executable
        mod_path = "lucid_memory.proxy_server:app" # Hardcoded for now
        port_str = str(self.config.get('local_proxy_port', 8000))

        try:
            # Prepare command
            cmd = [py_exe, "-m", "uvicorn", mod_path, "--host", "0.0.0.0", "--port", port_str, "--reload", "--log-level", "warning"]
            logging.info(f"Controller: Starting server with command: {' '.join(cmd)}")
            # Handle windows startup info
            startupinfo = None
            if os.name == 'nt': startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startupinfo.wShowWindow = subprocess.SW_HIDE
            # Launch
            self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', startupinfo=startupinfo)
            self._set_status(f"Status: Proxy server starting on port {port_str}...")
            # Return True indicating attempt was made (checking actual success requires polling later)
            return True
        except Exception as e: # Catch potential Popen errors
            logging.error(f"Controller: Failed to launch server process: {e}", exc_info=True)
            self.server_process = None
            self._set_status("Status: Error launching server!")
            return False

    def stop_server(self):
        """Stops the running Uvicorn server process."""
        proc = self.server_process
        if proc and proc.poll() is None: # Check if stored and running
            logging.info("Controller: Attempting to stop proxy server...")
            try:
                proc.terminate()
                try: proc.wait(timeout=3) # Wait briefly
                except subprocess.TimeoutExpired: logging.warning("Controller: Server kill timeout, killing."); proc.kill(); proc.wait()
                logging.info("Controller: Server stop command issued.")
                self._set_status("Status: Server stopped.")
            except Exception as e:
                logging.error(f"Controller: Error stopping server: {e}", exc_info=True)
                self._set_status("Status: Error stopping server.")
            finally:
                 self.server_process = None # Always clear handle after trying to stop
        else:
            logging.info("Controller: Stop server called, but server not running or handle lost.")
            self.server_process = None # Ensure handle is cleared
            # Update status only if seems relevant
            if "running" in self.last_status:
                 self._set_status("Status: Server not running.")


    def check_server_status(self):
        """Polls the server process status and updates internal state."""
        proc = self.server_process
        if not proc: return # No process to check

        return_code = proc.poll()

        if return_code is None: # Still running
             self._set_status(f"Status: Proxy server running on port {self.config.get('local_proxy_port', 8000)}")
        else: # Terminated
            if "running" in self.last_status: # Only log/alert if we thought it was running
                 stderr = "(Could not read stderr)"
                 stdout = "(Could not read stdout)"
                 try: stderr = "".join(proc.stderr.readlines()) # Attempt non-blocking read
                 except Exception: pass
                 try: stdout = "".join(proc.stdout.readlines())
                 except Exception: pass
                 logging.error(f"Controller: Server terminated unexpectedly (code {return_code}).\nStderr: {stderr}\nStdout: {stdout}")
                 self._set_status(f"Status: Server terminated unexpectedly (code {return_code})")
            else: # Already known to be stopped or errored
                 pass # Status likely already set by stop_server or previous check
            self.server_process = None # Clear handle

    # --- Optional: Add Chat Proxy Interaction ---
    # def send_chat_message(self, user_message: str, chat_callback: Callable[[str], None]):
    #    ... Implementation to send request to proxy and call chat_callback with result...
    #    ... Needs own background thread maybe ...