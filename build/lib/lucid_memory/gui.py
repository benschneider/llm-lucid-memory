import tkinter as tk
from tkinter import scrolledtext, filedialog, messagebox
import threading
import requests
import os
import json

# Configuration
CONFIG_PATH = "lucid_memory/proxy_config.json"

class LucidMemoryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lucid Memory - GUI")
        
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

        self.server_process = None

    def send_message(self, event=None):
        user_message = self.chat_entry.get()
        self.chat_entry.delete(0, tk.END)
        if user_message.strip() == "":
            return
        self.chat_display.insert(tk.END, f"User: {user_message}\n")
        
        # Send to proxy server
        try:
            with open(CONFIG_PATH, "r") as f:
                config = json.load(f)
            proxy_url = f"http://localhost:{config.get('local_proxy_port', 8000)}/chat"
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

    def load_context(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()
            # Future: send raw_text to digest and update memory graph
            self.memory_list.insert(tk.END, f"\nLoaded Context:\n{raw_text[:500]}...\n\n")

    def start_server(self):
        if self.server_process:
            messagebox.showinfo("Server", "Server already running!")
            return
        python_executable = os.sys.executable
        self.server_process = threading.Thread(target=lambda: os.system(f"{python_executable} -m uvicorn lucid_memory.proxy_server:app --reload"))
        self.server_process.start()
        messagebox.showinfo("Server", "Server launched!")

    def stop_server(self):
        messagebox.showinfo("Server", "Stopping the server is not yet implemented (work in progress).")

def main():
    root = tk.Tk()
    app = LucidMemoryApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()