import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import requests
import os
import json
import subprocess
from lucid_memory.digestor import Digestor
from lucid_memory.memory_graph import MemoryGraph

# Configuration
CONFIG_PATH = "lucid_memory/proxy_config.json"

class LucidMemoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lucid Memory - GUI")
        
        # Load config
        self.config = self.load_config()

        self.memory_graph = MemoryGraph()
        self.digestor = Digestor()

        # Top config editor
        config_frame = tk.Frame(root)
        config_frame.pack(pady=5)

        tk.Label(config_frame, text="Backend URL:").grid(row=0, column=0, sticky="w")
        self.backend_entry = tk.Entry(config_frame, width=50)
        self.backend_entry.grid(row=0, column=1)
        self.backend_entry.insert(0, self.config.get('backend_url', ''))

        tk.Label(config_frame, text="Model Name:").grid(row=1, column=0, sticky="w")
        self.model_entry = tk.Entry(config_frame, width=50)
        self.model_entry.grid(row=1, column=1)
        self.model_entry.insert(0, self.config.get('model_name', ''))

        tk.Label(config_frame, text="Local Port:").grid(row=2, column=0, sticky="w")
        self.port_entry = tk.Entry(config_frame, width=10)
        self.port_entry.grid(row=2, column=1, sticky="w")
        self.port_entry.insert(0, str(self.config.get('local_proxy_port', 8000)))

        tk.Button(config_frame, text="Save Config", command=self.save_config).grid(row=3, column=1, sticky="e", pady=5)
        
        # Frames
        self.chat_frame = tk.Frame(root)
        self.memory_frame = tk.Frame(root)
        self.control_frame = tk.Frame(root)
        
        self.chat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.memory_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Chat area
        self.chat_display = scrolledtext.ScrolledText(self.chat_frame, wrap=tk.WORD, height=20)
        self.chat_display.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        self.chat_entry = tk.Entry(self.chat_frame)
        self.chat_entry.pack(padx=10, pady=5, fill=tk.X)
        self.chat_entry.bind("<Return>", self.send_message)

        # Memory viewer
        self.memory_list = scrolledtext.ScrolledText(self.memory_frame, width=40, wrap=tk.WORD)
        self.memory_list.pack(padx=5, pady=5, fill=tk.BOTH, expand=True)
        
        # Control buttons
        tk.Button(self.control_frame, text="Load Context", command=self.load_context).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.control_frame, text="Start Server", command=self.start_server).pack(side=tk.LEFT, padx=5, pady=5)
        tk.Button(self.control_frame, text="Stop Server", command=self.stop_server).pack(side=tk.LEFT, padx=5, pady=5)

        self.status_label = tk.Label(self.control_frame, text="")
        self.status_label.pack(side=tk.LEFT, padx=5)

        self.server_process = None

    def load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        else:
            return {
                "backend_url": "http://localhost:11434/v1/chat/completions",
                "model_name": "mistral",
                "local_proxy_port": 8000
            }

    def save_config(self):
            new_config = {
                "backend_url": self.backend_entry.get(),
                "model_name": self.model_entry.get(),
                "local_proxy_port": int(self.port_entry.get())
            }
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(new_config, f, indent=2)
            messagebox.showinfo("Saved", "Configuration updated successfully.")
            self.config = new_config

    def send_message(self, event=None):
        user_message = self.chat_entry.get()
        self.chat_entry.delete(0, tk.END)
        if user_message.strip() == "":
            return
        self.chat_display.insert(tk.END, f"User: {user_message}\n")
        
        # Send to proxy server
        try:
            proxy_url = f"http://localhost:{self.config.get('local_proxy_port', 8000)}/chat"
            payload = {
                "messages": [{"role": "user", "content": user_message}],
                "temperature": 0.2
            }
            response = requests.post(proxy_url, json=payload)
            data = response.json()
            assistant_reply = data["choices"][0]["message"]["content"]
            self.chat_display.insert(tk.END, f"Assistant: {assistant_reply}\n\n")
        except Exception as e:
            self.chat_display.insert(tk.END, f"Error: {str(e)}\n")

    def load_context(self, event=None):
        filename = filedialog.askopenfilename(
            title="Select Context File",
            filetypes=(("Text files", "*.txt"), ("All files", "*.*"))
        )
        if filename:
            with open(filename, "r", encoding="utf-8") as f:
                raw_text = f.read()

            self.status_label.config(text="Digesting...")
            self.control_frame.update_idletasks()

            digestor = Digestor()

            node = digestor.digest(raw_text, node_id=f"manual_{os.path.basename(filename)}")
            self.memory_graph.add_node(node)
            self.refresh_memory_display()
            self.status_label.config(text="Done.")

            messagebox.showinfo("Context Loaded", f"Loaded and digested: {os.path.basename(filename)}")

    def refresh_memory_display(self):
        self.memory_list.delete(1.0, tk.END)
        for node in self.memory_graph.nodes.values():
            self.memory_list.insert(tk.END, f"ID: {node.id}\n")
            self.memory_list.insert(tk.END, f"Summary: {node.summary}\n\n")
        self.memory_list.see(tk.END)  # Auto-scroll to bottom after refresh

    def start_server(self):
        if self.server_process:
            messagebox.showinfo("Server", "Server already running!")
            return
        python_executable = os.sys.executable
        self.server_process = subprocess.Popen(
            [python_executable, "-m", "uvicorn", "lucid_memory.proxy_server:app", "--reload"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        messagebox.showinfo("Server", "Server launched!")

    def stop_server(self):
        if self.server_process:
            self.server_process.terminate()
            self.server_process = None
            messagebox.showinfo("Server", "Server stopped!")
        else:
            messagebox.showinfo("Server", "No server running.")

    

def main():
    root = tk.Tk()
    app = LucidMemoryApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()