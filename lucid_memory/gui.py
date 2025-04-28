# lucid_memory/gui.py
# v0.2.6 - Refactored with Controller

import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import requests # Keep requests mainly for chat sending if not moved to controller
import os
import subprocess
import threading
import logging
import json # Needed for get/save config debug perhaps
from typing import List, Dict, Any, Optional

# Import Controller and Core components needed for display/hints
from lucid_memory.controller import LucidController
from lucid_memory.memory_node import MemoryNode

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Constants like MEMORY_GRAPH_PATH might not be needed here anymore

# --- Main Application Class ---
class LucidMemoryApp:
    def __init__(self, root):
        """Initializes the GUI application and the backend Controller."""
        self.root = root
        self.root.title("Lucid Memory - GUI v0.2.6 (Refactored)")
        self.root.minsize(700, 550) # Adjusted min size slightly

        # Create controller instance
        # Handle potential controller init errors gracefully
        try:
            self.controller = LucidController()
        except Exception as e:
             logging.error(f"CRITICAL: Failed to initialize LucidController: {e}", exc_info=True)
             messagebox.showerror("Startup Error", f"Failed to initialize controller. Check logs.\n{e}")
             self.root.destroy() # Cannot proceed without controller
             return

        # Set callbacks for the controller
        self.controller.status_update_callback = self._update_status_label
        self.controller.graph_update_callback = self.refresh_memory_display # Direct refresh for now

        self._build_ui() # Now build UI based on controller state

        # Set initial status and potentially trigger first server check
        self._update_status_label(self.controller.get_last_status())
        self.root.after(2500, self.controller.check_server_status) # Initial check server

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        logging.info("GUI Initialized.")

    # --- UI Building Helpers ---
    # (These methods remain similar, but get initial data from controller)
    def _build_ui(self):
        self._build_config_frame()
        self._build_main_frames()
        self._build_control_frame()
        self.refresh_memory_display() # Initial population

    def _build_config_frame(self):
        frame = tk.LabelFrame(self.root, text="Configuration")
        frame.pack(pady=5, padx=10, fill="x")
        current_config = self.controller.get_config() # Get config from controller

        tk.Label(frame, text="Backend URL:").grid(row=0, column=0, sticky="w", padx=5)
        self.backend_entry = tk.Entry(frame, width=60)
        self.backend_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.backend_entry.insert(0, current_config.get('backend_url', ''))

        tk.Label(frame, text="Model Name:").grid(row=1, column=0, sticky="w", padx=5)
        self.model_entry = tk.Entry(frame, width=60)
        self.model_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.model_entry.insert(0, current_config.get('model_name', ''))

        tk.Label(frame, text="Local Port:").grid(row=2, column=0, sticky="w", padx=5)
        self.port_entry = tk.Entry(frame, width=10)
        self.port_entry.grid(row=2, column=1, padx=(5,0), pady=2, sticky="w")
        self.port_entry.insert(0, str(current_config.get('local_proxy_port', 8000)))

        tk.Button(frame, text="Save Config", command=self.action_save_config).grid(row=3, column=1, sticky="e", pady=5, padx=5)
        frame.columnconfigure(1, weight=1)

    def _build_main_frames(self):
        main_frame = tk.Frame(self.root); main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self._build_chat_frame(main_frame); self._build_memory_frame(main_frame)

    def _build_chat_frame(self, parent):
        frame = tk.LabelFrame(parent, text="Chat"); frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        self.chat_display = scrolledtext.ScrolledText(frame, wrap=tk.WORD, height=15, state=tk.DISABLED); self.chat_display.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.chat_entry = tk.Entry(frame); self.chat_entry.pack(padx=5, pady=(0,5), fill=tk.X); self.chat_entry.bind("<Return>", self.action_send_message)

    def _build_memory_frame(self, parent):
        frame = tk.LabelFrame(parent, text="Memory Nodes"); frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        self.memory_list = scrolledtext.ScrolledText(frame, width=55, wrap=tk.WORD, state=tk.DISABLED); self.memory_list.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)

    def _build_control_frame(self):
        frame = tk.Frame(self.root); frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)
        # Link buttons to action methods
        tk.Button(frame, text="Load Context File", command=self.action_load_context).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Start Proxy Server", command=self.action_start_server).pack(side=tk.LEFT, padx=5)
        tk.Button(frame, text="Stop Proxy Server", command=self.action_stop_server).pack(side=tk.LEFT, padx=5)
        self.status_label = tk.Label(frame, text="Status: Initializing...", relief=tk.SUNKEN, anchor="w"); self.status_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)


    # --- UI Update Callbacks ---

    def _update_status_label(self, text: str):
        """Thread-safe method to update the status bar label."""
        self.root.after(0, lambda: self.status_label.config(text=text))

    def _append_chat_message(self, text: str):
         """Thread-safe method to append text to the chat display."""
         def append():
             # Temporarily enable, insert, scroll, disable
             try:
                 self.chat_display.config(state=tk.NORMAL)
                 self.chat_display.insert(tk.END, text)
                 self.chat_display.see(tk.END)
             except tk.TclError as e:
                 logging.warning(f"Error updating chat display (maybe closed?): {e}")
             finally: # Ensure it's always disabled
                  try: self.chat_display.config(state=tk.DISABLED)
                  except tk.TclError: pass # Ignore if window closed
         self.root.after(0, append)


    # --- Actions (Called by UI Events, Delegate to Controller) ---

    def action_save_config(self):
        """Handles the Save Config button click."""
        logging.info("GUI: Save Config action initiated.")
        try:
            new_config_values = {
                "backend_url": self.backend_entry.get().strip(),
                "model_name": self.model_entry.get().strip(),
                "local_proxy_port": int(self.port_entry.get()) # Validate int here
            }
            if self.controller.update_config(new_config_values):
                 messagebox.showinfo("Config Saved", "Configuration updated and saved.")
            else:
                 messagebox.showerror("Config Error", "Failed to save configuration. Check inputs and logs.")
        except ValueError:
            messagebox.showerror("Input Error", "Local port must be a valid integer.")
        except Exception as e: # Catch unexpected errors during update
            messagebox.showerror("Error", f"An unexpected error occurred: {e}")
            logging.error("Unexpected error during save config action.", exc_info=True)

    def action_load_context(self):
        """Handles the Load Context File button."""
        logging.info("GUI: Load Context action initiated.")
        if not self.controller.is_digestor_ready():
            messagebox.showerror("Error", "Digestor is not ready. Please check configuration."); return
        if self.controller.is_processing():
            messagebox.showwarning("Busy", "Currently processing another file. Please wait."); return

        filename = filedialog.askopenfilename(
            title="Select Context File",
            filetypes=(("Python","*.py"),("Markdown","*.md"),("Text","*.txt"),("All","*.*"))
        )
        if filename:
            logging.info(f"GUI: User selected file: {filename}")
            # Delegate to controller - controller handles threading
            self.controller.start_digestion_for_file(filename)
        else:
            logging.info("GUI: File selection cancelled.")

    def action_start_server(self):
        """Handles the Start Proxy Server button."""
        logging.info("GUI: Start Server action initiated.")
        if self.controller.start_server():
             # Optional: Give a little feedback, status will update via check_server_status
             # messagebox.showinfo("Server", "Attempting to start server...")
             self.root.after(2000, self.controller.check_server_status) # Schedule a check
        else:
              # Controller already logged the error and updated status
              messagebox.showerror("Server Error", "Failed to start server process. Check logs.")

    def action_stop_server(self):
        """Handles the Stop Proxy Server button."""
        logging.info("GUI: Stop Server action initiated.")
        self.controller.stop_server()
        # Controller updates status via callback

    def action_send_message(self, event=None): # Renamed from send_message
        """Handles sending message from chat entry."""
        user_message = self.chat_entry.get().strip()
        if not user_message: return
        self.chat_entry.delete(0, tk.END)
        self._append_chat_message(f"User: {user_message}\n")

        # --- Simplification: Use Requests Directly For Now ---
        # Interaction with the proxy *could* be moved to the controller,
        # but keeping it here is simpler initially for chat feedback.
        if not self.controller.server_process or self.controller.server_process.poll() is not None:
             self._append_chat_message("Error: Proxy server not running.\n\n")
             return

        proxy_url = f"http://localhost:{self.controller.config.get('local_proxy_port', 8000)}/chat"
        payload = { "messages": [{"role": "user", "content": user_message}], "temperature": 0.2 }
        # Run chat request in a separate thread as before
        threading.Thread(target=self._send_chat_request_thread, args=(proxy_url, payload), daemon=True).start()

    def _send_chat_request_thread(self, url, payload): # Renamed from _send_request_thread
        """Sends chat HTTP request in background. Updates chat UI."""
        # This function contains the actual 'requests' call
        try:
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            reply = "Error: Could not parse chat response."
            # Add more defensive checks for response structure
            if isinstance(data.get("choices"), list) and len(data["choices"]) > 0:
                 msg = data["choices"][0].get("message")
                 if isinstance(msg, dict): reply = msg.get("content", reply)
            self._append_chat_message(f"Assistant: {reply}\n\n")
        # Specific exception clauses are good here
        except requests.exceptions.ConnectionError: self._append_chat_message(f"Error: Could not connect to proxy at {url}.\n\n")
        except requests.exceptions.Timeout: self._append_chat_message("Error: Chat request timed out.\n\n")
        except requests.exceptions.RequestException as e: logging.warning(f"Chat proxy error: {e}"); self._append_chat_message(f"Error chatting with proxy: {e}\n\n")
        except Exception as e: logging.error("Process chat response error", exc_info=True); self._append_chat_message(f"Error processing chat response: {e}\n\n")


    # --- Display Refresh ---
    def refresh_memory_display(self):
        """Pulls nodes from controller and refreshes memory list display."""
        logging.debug("GUI: Refreshing memory display.")
        # Keep broad try-except as UI updates can have strange errors sometimes
        try:
            self.memory_list.config(state=tk.NORMAL)
            self.memory_list.delete(1.0, tk.END)
            nodes = self.controller.get_memory_nodes() # Get data from controller
            if not nodes:
                self.memory_list.insert(tk.END, "(Memory graph is empty)")
            else:
                # Sort nodes here for display
                sorted_nodes = sorted(nodes.items(), key=lambda item: (getattr(item[1], 'sequence_index', float('inf')), item[0]))
                display_lines = [] # Build as list first
                for node_id, node in sorted_nodes:
                    lines = []
                    lines.append(f"ID: {node.id}")
                    # -- Linking Info --
                    seq = getattr(node, 'sequence_index', None)
                    parent = getattr(node, 'parent_identifier', None)
                    if seq is not None: lines.append(f"Seq: {seq}")
                    if parent: lines.append(f"Parent: {parent}")
                    # -- Core Digested Info --
                    lines.append(f"Summary: {node.summary}")
                    lines.append(f"Tags: {', '.join(node.tags) if node.tags else '(None)'}")
                    # -- Concepts/Steps --
                    concepts = getattr(node, 'key_concepts', []) # Use actual field name
                    lines.append(f"Key Concepts ({len(concepts)}):")
                    lines.extend([f"  - {c}" for c in concepts] if concepts else ["  (None extracted)"])
                    # -- Dependencies/Outputs (Only if present) --
                    deps = getattr(node, 'dependencies', [])
                    outs = getattr(node, 'produced_outputs', [])
                    if deps or outs:
                        lines.append(f"Dependencies ({len(deps)}):")
                        lines.extend([f"  -> {d}" for d in deps] if deps else ["  (None detected)"])
                        lines.append(f"Produced Outputs ({len(outs)}):")
                        lines.extend([f"  <- {o}" for o in outs] if outs else ["  (None detected)"])
                    lines.append("-" * 20)
                    display_lines.append("\n".join(lines) + "\n\n") # Join lines for one node
                # Insert the whole text at once - potentially faster for large lists
                self.memory_list.insert(tk.END, "".join(display_lines))

            self.memory_list.see(tk.END)
        except Exception as e:
            logging.error(f"Refresh display error: {e}", exc_info=True)
            try: self.memory_list.insert(tk.END, f"\n--- ERROR REFRESHING ---\n")
            except Exception: pass # Ignore if error display fails too
        finally:
            self.memory_list.config(state=tk.DISABLED)


    # --- Window Closing ---
    def on_closing(self):
        """Handles window close event."""
        logging.info("GUI: Closing event triggered.")
        # Delegate server stop cleanly
        self.controller.stop_server() # Ask controller to stop if running
        # Add a small delay to allow server process to potentially terminate
        # Or better: track server_process state via controller after stop call
        # For now, assume stop_server handles waiting/killing
        self.root.destroy()

# --- Main Execution ---
def main():
    root = tk.Tk()
    app = LucidMemoryApp(root) # Create the application instance
    root.mainloop()

if __name__ == "__main__":
    main()