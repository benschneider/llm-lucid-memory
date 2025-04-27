import tkinter as tk
from tkinter import messagebox
import json
import subprocess
import os
import signal

CONFIG_PATH = "lucid_memory/proxy_config.json"
server_process = None  # Global server process handle

def save_config(backend_url, model_name, local_port):
    config = {
        "backend_url": backend_url,
        "model_name": model_name,
        "local_proxy_port": int(local_port)
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    messagebox.showinfo("Saved", "Configuration saved successfully!")

def launch_server(status_label):
    global server_process
    if server_process:
        messagebox.showinfo("Server", "Server already running!")
        return

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
            port = config.get("local_proxy_port", 8000)
    else:
        port = 8000

    python_executable = os.sys.executable
    server_process = subprocess.Popen(
        [python_executable, "-m", "uvicorn", "lucid_memory.proxy_server:app",
         "--host", "127.0.0.1", "--port", str(port), "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    def read_output():
        for line in server_process.stdout:
            print(line, end="")
        for line in server_process.stderr:
            print(line, end="")

    import threading
    threading.Thread(target=read_output, daemon=True).start()

    status_label.config(text=f"Server running on port {port}")

def stop_server(status_label):
    global server_process
    if server_process:
        server_process.terminate()
        server_process = None
        status_label.config(text="Server stopped")
        messagebox.showinfo("Server", "Proxy server stopped.")
    else:
        messagebox.showinfo("Server", "No server is currently running.")

def main():
    window = tk.Tk()
    window.title("LLM Lucid Memory - Start Server")

    try:
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    except Exception:
        config = {
            "backend_url": "http://localhost:11434/v1/chat/completions",
            "model_name": "mistral",
            "local_proxy_port": 8000
        }

    tk.Label(window, text="Backend URL (e.g., http://localhost:11434/v1/chat/completions)").pack()
    backend_entry = tk.Entry(window, width=50)
    backend_entry.pack()
    backend_entry.insert(0, config.get("backend_url", ""))

    tk.Label(window, text="Model Name (e.g., mistral, phi)").pack()
    model_entry = tk.Entry(window, width=50)
    model_entry.pack()
    model_entry.insert(0, config.get("model_name", ""))

    tk.Label(window, text="Local Proxy Port (default 8000)").pack()
    port_entry = tk.Entry(window, width=10)
    port_entry.pack()
    port_entry.insert(0, str(config.get("local_proxy_port", 8000)))

    tk.Button(window, text="Save Config", command=lambda: save_config(
        backend_entry.get(), model_entry.get(), port_entry.get()
    )).pack(pady=5)

    status_label = tk.Label(window, text="Server not running")
    status_label.pack(pady=5)

    tk.Button(window, text="Launch Proxy Server", command=lambda: launch_server(status_label)).pack(pady=5)
    tk.Button(window, text="Stop Proxy Server", command=lambda: stop_server(status_label)).pack(pady=5)

    tk.Label(window, text="\nThis server bridges your app with the LLM backend using smart memory!").pack()

    window.mainloop()

if __name__ == "__main__":
    main()