# lucid_memory/gui.py
# v0.2.5 - Indentation Fix & Cleanup

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import requests
import os
import json
import subprocess
import threading
import logging
import re
from typing import List, Dict, Any, Optional

from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.chunker import chunk_file
from lucid_memory.digestor import Digestor
from lucid_memory.processor import ChunkProcessor

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "proxy_config.json")
MEMORY_GRAPH_PATH = "memory_graph.json"
DEFAULT_CONFIG = {
    "backend_url": "http://localhost:11434/v1/chat/completions",
    "model_name": "mistral",
    "local_proxy_port": 8000
}

# --- Main Application Class ---
class LucidMemoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lucid Memory - GUI v0.2.5")
        self.config = self._load_config()
        self.memory_graph = self._load_memory_graph()
        self.digestor_ready = self._check_digestor_readiness()
        self.server_process = None
        self.processor_thread: Optional[threading.Thread] = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        logging.info("GUI Initialized.")
        if not self.digestor_ready:
             self.root.after(100, lambda: self._update_status_label("Status: Warning - Digestor config invalid."))

    # --- Initialization Helpers ---
    def _load_config(self):
        """Loads configuration from JSON file or returns default."""
        cfg = DEFAULT_CONFIG.copy() # Start with defaults
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    loaded_cfg = json.load(f)
                # Update defaults with loaded values, keeping defaults for missing keys
                cfg.update(loaded_cfg)
                logging.info(f"Loaded configuration from {CONFIG_PATH}")
            except (json.JSONDecodeError, Exception) as e:
                 logging.error(f"Config Error parsing {CONFIG_PATH}: {e}. Using defaults.", exc_info=False)
                 # Keep the default cfg
        else:
            logging.warning(f"Config file {CONFIG_PATH} not found. Creating default.")
            try:
                os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
                with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                    json.dump(cfg, f, indent=2) # Save the default cfg
            except Exception as e:
                logging.error(f"Config Error creating default file: {e}", exc_info=False)
        return cfg

    def _load_memory_graph(self):
        """Loads the memory graph from the shared JSON file."""
        graph = MemoryGraph()
        if os.path.exists(MEMORY_GRAPH_PATH):
            try:
                graph.load_from_json(MEMORY_GRAPH_PATH)
                logging.info(f"Loaded {len(graph.nodes)} nodes from {MEMORY_GRAPH_PATH}")
            except Exception as e: # Keep broad except here for file load robustness
                logging.warning(f"Memory Load Warning parsing {MEMORY_GRAPH_PATH}: {e}. Starting empty.", exc_info=False)
        else:
             logging.info(f"Memory file {MEMORY_GRAPH_PATH} not found. Starting empty.")
        return graph

    def _check_digestor_readiness(self) -> bool:
        """Checks if Digestor can be initialized based on config."""
        if not self.config.get('backend_url') or not self.config.get('model_name'):
            logging.error("Digestor readiness check: Missing backend_url or model_name.")
            return False
        try:
            Digestor() # This might raise errors if prompts.yaml is missing/invalid
            logging.info("Digestor readiness check passed.")
            return True
        except Exception as e: # Catch init errors (e.g., file not found)
            logging.error(f"Digestor readiness check failed: {e}", exc_info=True)
            return False

    # --- UI Building ---
    def _build_ui(self):
        """Builds the main UI elements."""
        self._build_config_frame()
        self._build_main_frames()
        self._build_control_frame()
        self.refresh_memory_display()

    def _build_config_frame(self):
        frame = tk.LabelFrame(self.root, text="Configuration")
        frame.pack(pady=5, padx=10, fill="x")

        tk.Label(frame, text="Backend URL:").grid(row=0, column=0, sticky="w", padx=5)
        self.backend_entry = tk.Entry(frame, width=60)
        self.backend_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.backend_entry.insert(0, self.config.get('backend_url', ''))

        tk.Label(frame, text="Model Name:").grid(row=1, column=0, sticky="w", padx=5)
        self.model_entry = tk.Entry(frame, width=60)
        self.model_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.model_entry.insert(0, self.config.get('model_name', ''))

        tk.Label(frame, text="Local Port:").grid(row=2, column=0, sticky="w", padx=5)
        self.port_entry = tk.Entry(frame, width=10)
        self.port_entry.grid(row=2, column=1, padx=(5,0), pady=2, sticky="w")
        self.port_entry.insert(0, str(self.config.get('local_proxy_port', 8000)))

        tk.Button(frame, text="Save Config", command=self.save_config).grid(row=3, column=1, sticky="e", pady=5, padx=5)
        frame.columnconfigure(1, weight=1)

    def _build_main_frames(self):
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._build_chat_frame(main_frame)
        self._build_memory_frame(main_frame)

    def _build_chat_frame(self, parent):
        frame = tk.LabelFrame(parent, text="Chat")
        frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.chat_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15, state=tk.DISABLED)
        self.chat_display.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.chat_entry = tk.Entry(frame)
        self.chat_entry.pack(padx=5, pady=(0,5), fill=tk.X)
        self.chat_entry.bind("<Return>", self.send_message)

    def _build_memory_frame(self, parent):
        frame = tk.LabelFrame(parent, text="Memory Nodes")
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.memory_list = scrolledtext.ScrolledText(frame, width=55, wrap=tk.WORD, state=tk.DISABLED)
        self.memory_list.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def _build_control_frame(self):
        frame = tk.Frame(self.root)
        frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        tk.Button(frame, text="Load Context File", command=self.load_context_file).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Start Proxy Server", command=self.start_server).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Stop Proxy Server", command=self.stop_server).pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(frame, text="Status: Initialized", relief=tk.SUNKEN, anchor="w")
        self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

    # --- UI Update Helpers (Callbacks) ---
    def _update_status_label(self, text):
        """Safely updates status label via root.after()."""
        self.root.after(0, lambda: self.status_label.config(text=text))

    def _handle_processing_completion(self, graph_changed: bool):
        """Callback when chunk processing finishes."""
        logging.info(f"Processing completion callback. Graph changed: {graph_changed}")
        if graph_changed:
            self.root.after(0, self.refresh_memory_display)
        self.processor_thread = None # Clear thread tracker

    def _append_chat_message(self, text):
         """Safely appends text to chat display via root.after()."""
         def append():
             self.chat_display.config(state=tk.NORMAL)
             self.chat_display.insert(tk.END, text)
             self.chat_display.config(state=tk.DISABLED)
             self.chat_display.see(tk.END)
         self.root.after(0, append)

    # --- Core Functionality Methods ---
    def save_config(self):
        """Saves current configuration."""
        try:
            new_port = int(self.port_entry.get())
            new_config = {
                "backend_url": self.backend_entry.get().strip(),
                "model_name": self.model_entry.get().strip(),
                "local_proxy_port": new_port
            }
            if not new_config['backend_url'] or not new_config['model_name']:
                 raise ValueError("Backend URL/Model Name required.")

            # Keep try-except for file operations
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2)

            messagebox.showinfo("Saved", "Configuration updated.")
            self.config = new_config
            self.digestor_ready = self._check_digestor_readiness()
            status = "Status: Config saved. Digestor " + ("ready." if self.digestor_ready else "NOT ready.")
            self._update_status_label(status)
        except ValueError as e:
             messagebox.showerror("Input Error", f"{e}") # More specific error
        except Exception as e: # Keep broad except for file save
             messagebox.showerror("Save Error", f"Failed to save configuration: {e}")
             logging.error("Save config error", exc_info=True)

    def send_message(self, event=None):
        """Handles sending a message from the chat entry."""
        user_message = self.chat_entry.get().strip()
        if not user_message: return
        self.chat_entry.delete(0, tk.END)
        self._append_chat_message(f"User: {user_message}\n")

        if not self.server_process or self.server_process.poll() is not None:
             self._append_chat_message("Error: Proxy server not running.\n\n")
             return

        proxy_url = f"http://localhost:{self.config.get('local_proxy_port', 8000)}/chat"
        payload = { "messages": [{"role": "user", "content": user_message}], "temperature": 0.2 }
        # Keep thread for network request to avoid blocking GUI
        threading.Thread(target=self._send_request_thread, args=(proxy_url, payload), daemon=True).start()

    def _send_request_thread(self, url, payload):
        """Sends HTTP request in background. Updates chat UI."""
        try: # Keep try-except for network/request errors
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status() # Check for HTTP errors 4xx/5xx
            data = response.json()
            reply = "Error: Could not parse LLM response."
            # Simplify response checking slightly
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                 msg = choices[0].get("message")
                 reply = msg.get("content", reply) if isinstance(msg, dict) else reply
            self._append_chat_message(f"Assistant: {reply}\n\n")
        except requests.exceptions.Timeout:
            self._append_chat_message("Error: Chat request timed out.\n\n")
        except requests.exceptions.RequestException as e: # Catch all request-related errors
            logging.warning(f"Proxy comm error: {e}", exc_info=False)
            self._append_chat_message(f"Error communicating with proxy: {e}\n\n")
        except Exception as e: # Catch potential JSON errors or others
            logging.error("Process chat response error", exc_info=True)
            self._append_chat_message(f"Error processing response: {e}\n\n")

    # --- Context Loading Trigger ---
    def load_context_file(self):
        """Opens file dialog, chunks file, and starts Processor in background thread."""
        if not self.digestor_ready:
            messagebox.showerror("Error", "Digestor not ready. Check config."); return
        if self.processor_thread and self.processor_thread.is_alive():
             messagebox.showwarning("Busy", "Already processing file."); return

        filename = filedialog.askopenfilename(
            title="Select Context File",
            filetypes=(("Python","*.py"),("Markdown","*.md"),("Text","*.txt"),("All","*.*"))
        )
        if not filename: return

        try: # Keep try-except for file reading/chunking
            with open(filename, "r", encoding="utf-8", errors='ignore') as f:
                 raw_text = f.read()

            logging.info(f"Starting chunking for {filename}")
            self._update_status_label(f"Status: Chunking {os.path.basename(filename)}...")
            self.root.update_idletasks()
            chunks = chunk_file(filename, raw_text) # External call, might fail
            if not chunks:
                 messagebox.showwarning("Chunking", "File yielded no chunks."); self._update_status_label("Status: File not chunked."); return

            logging.info(f"Chunking complete ({len(chunks)} chunks). Initializing processor...")
            self._update_status_label(f"Status: Chunked ({len(chunks)}). Starting digestion...")
            self.root.update_idletasks()

            # Keep try-except for critical Digestor init
            try: current_digestor = Digestor()
            except Exception as e: logging.error(f"Failed Digestor init: {e}", exc_info=True); messagebox.showerror("Error", f"Digestor init failed: {e}"); self._update_status_label("Status: Error - Failed Digestor init."); return

            processor = ChunkProcessor( digestor=current_digestor, memory_graph=self.memory_graph, status_callback=self._update_status_label, completion_callback=self._handle_processing_completion )
            self.processor_thread = threading.Thread( target=processor.process_chunks, args=(chunks, os.path.basename(filename)), daemon=True )
            self.processor_thread.start()

        except Exception as e: # Catch file read or chunk_file errors
            logging.error(f"File Read/Chunk Error: {e}", exc_info=True)
            messagebox.showerror("File Load Error", f"Read/chunk failed: {e}")
            self._update_status_label("Status: Error reading/chunking file")

    # --- Display Refresh ---
    # Corrected Indentation
    def refresh_memory_display(self):
        """Clears and re-populates the memory list. MUST be called from main thread."""
        # Keep broad try-except as UI updates can fail unexpectedly
        try:
            self.memory_list.config(state=tk.NORMAL)
            self.memory_list.delete(1.0, tk.END)
            if not self.memory_graph.nodes:
                self.memory_list.insert(tk.END, "(Memory graph is empty)")
            else:
                sorted_nodes = sorted(self.memory_graph.nodes.items())
                for node_id, node in sorted_nodes:
                    display_text = f"ID: {node.id}\n"
                    metadata = getattr(node, 'source_chunk_metadata', None)
                    if metadata and metadata.get('identifier'):
                        display_text += f"Source ID: {metadata['identifier']}\n"
                    display_text += f"Summary: {node.summary}\n"
                    display_text += f"Tags: {', '.join(node.tags) if node.tags else '(None)'}\n"
                    concepts_or_steps = getattr(node, 'key_concepts', getattr(node, 'logical_steps', []))
                    concepts_label = "Concepts" if hasattr(node, 'key_concepts') else "Logical Steps"
                    display_text += f"{concepts_label} ({len(concepts_or_steps)}):\n"
                    display_text += "".join([f"  - {item}\n" for item in concepts_or_steps]) if concepts_or_steps else "  (None extracted)\n"
                    display_text += f"Follow-up ({len(node.follow_up_questions)}):\n"
                    display_text += "".join([f"  ? {q}\n" for q in node.follow_up_questions]) if node.follow_up_questions else "  (None)\n"
                    display_text += "-" * 20 + "\n\n"
                    self.memory_list.insert(tk.END, display_text)
            self.memory_list.see(tk.END)
        except Exception as e:
            logging.error(f"Refresh display error: {e}", exc_info=True)
            try: # Nested try to insert error message is acceptable here
                self.memory_list.insert(tk.END, f"\n--- ERROR REFRESHING DISPLAY ---\n{e}\n")
            except Exception: pass # Ignore if error display fails
        finally:
            self.memory_list.config(state=tk.DISABLED)


    # --- Server Control ---
    def start_server(self):
        """Starts the Uvicorn proxy server subprocess."""
        if self.server_process and self.server_process.poll() is None:
            messagebox.showinfo("Server", "Already running."); return
        py_exe = os.sys.executable; mod_path = "lucid_memory.proxy_server:app"
        try: # Keep try-except for subprocess/port errors
            port_str = self.port_entry.get(); int(port_str) # Validate
            cmd = [py_exe, "-m", "uvicorn", mod_path, "--host", "0.0.0.0", "--port", port_str, "--reload", "--log-level", "warning"]
            logging.info(f"Starting server: {' '.join(cmd)}"); startupinfo = None
            if os.name == 'nt': startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW; startupinfo.wShowWindow = subprocess.SW_HIDE
            # Use text=True for easier stdout/stderr handling
            self.server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', startupinfo=startupinfo)
            self._update_status_label(f"Status: Proxy server starting on port {port_str}...")
            self.root.after(2000, self.check_server_status) # Check later
        except ValueError: messagebox.showerror("Error", "Invalid port number.")
        except Exception as e: logging.error(f"Server launch fail: {e}", exc_info=True); messagebox.showerror("Server Error", f"Failed: {e}"); self._update_status_label("Status: Failed start")

    def check_server_status(self):
        """Checks if the server process is running or terminated."""
        if not self.server_process: return
        rc = self.server_process.poll()
        if rc is not None: # Server terminated
            # Simplified error reading (might miss output if read too late)
            stderr = self.server_process.stderr.read() if self.server_process.stderr else "(No stderr)"
            stdout = self.server_process.stdout.read() if self.server_process.stdout else "(No stdout)"
            logging.error(f"Server terminated (code {rc}).\nStderr: {stderr}\nStdout: {stdout}")
            messagebox.showerror("Server Error", f"Server terminated (code {rc}). Check logs.")
            self._update_status_label("Status: Server terminated unexpectedly")
            self.server_process = None
        else: # Server still running
            self._update_status_label(f"Status: Proxy server running on port {self.port_entry.get()}")

    def stop_server(self):
        """Stops the running Uvicorn server process."""
        if self.server_process and self.server_process.poll() is None:
            logging.info("Stopping server...");
            try: # Keep try-except for process termination
                self.server_process.terminate()
                try: self.server_process.wait(timeout=3) # Shorter wait
                except subprocess.TimeoutExpired: logging.warning("Killing server."); self.server_process.kill(); self.server_process.wait()
                logging.info("Server stopped."); messagebox.showinfo("Server", "Server stopped.")
                self._update_status_label("Status: Stopped")
            except Exception as e: logging.error(f"Stop server error: {e}", exc_info=True); messagebox.showerror("Server Error", f"Stop error: {e}"); self._update_status_label("Status: Error stopping")
            finally: self.server_process = None
        else:
            logging.info("Stop called, server not running.")
            self._update_status_label("Status: Idle (Server not running)") if "running" in self.status_label.cget("text") else None

    def on_closing(self):
        """Handles window close event."""
        if self.server_process and self.server_process.poll() is None:
            if messagebox.askokcancel("Quit", "Server running. Stop and exit?"):
                self.stop_server()
                self.root.destroy()
        else:
            self.root.destroy()

# --- Main Execution ---
def main():
    root = tk.Tk()
    root.minsize(700, 500)
    app = LucidMemoryApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()