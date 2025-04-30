# lucid_memory/managers/server_manager.py
import os
import sys
import subprocess
import logging
import time
from typing import Optional, Tuple, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ServerManager - %(message)s')

class ServerManager:
    """Manages the lifecycle of the Uvicorn proxy server subprocess."""
    def __init__(self, config_manager): # Depends on config for port
        self.config_mgr = config_manager
        self.server_process: Optional[subprocess.Popen] = None
        self.last_known_port: Optional[int] = None # Track port used

    def _get_current_port(self) -> int:
        # Safely retrieve port from config, use fallback
         return self.config_mgr.get_value('local_proxy_port', 8000) # Uses Safe GET from config manager..

    def start_server(self) -> bool:
        """Starts the Uvicorn server subprocess."""
        if self.is_running():
            pid = self.server_process.pid if self.server_process else '?'
            logging.warning(f"Start server ignored - already running (PID: {pid}, Port: {self.last_known_port}).")
            return True # Indicate running state

        port = self._get_current_port()
        py_exe = sys.executable
        app_module = "lucid_memory.proxy_server:app" # Assuming path fixed relative to package root ok..
        # --log-level info Useful Starting point.. Maybe make config option later?? ok.
        cmd = [py_exe, "-m", "uvicorn", app_module, "--host", "0.0.0.0", "--port", str(port), "--log-level", "info"]
        # NO reload default --> Safer Prod run behavior default.. OK.

        logging.info(f"Attempting START server: {' '.join(cmd)}")
        try:
            startupinfo = None # Windows suppress console
            if os.name == 'nt': startupinfo = subprocess.STARTUPINFO(); startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW;

            # LAUNCH..! Capture OUT/ERR streams useful debugging failures.!
            self.server_process = subprocess.Popen(cmd,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, # Caputre ok..
                                    text=True, encoding='utf-8', errors='replace', # Text mode + safer decode ok..
                                    startupinfo=startupinfo)
            self.last_known_port = port # Store the port we TRIED to launch on..
            logging.info(f"Server process started (PID={self.server_process.pid}). Waiting briefly...")
            time.sleep(0.5) # Tiny pause maybe allows better immediate status check? Ok..
            return True

        except FileNotFoundError:
             logging.critical("Uvicorn/Python NOT FOUND? Ensure Python Env correct + FastAPI deps install OK!.")
             self.server_process=None ; return False # Cannot Start -> Return Error State..
        except Exception as e:
            logging.exception(f"Server launch FAILED: {e}")
            self.server_process = None ; return False # Critical startup Failed -> Return Error State.

    def stop_server(self) -> bool:
        """Stops the running Uvicorn server process (if any). Returns True if stop attempt seemed ok."""
        proc = self.server_process
        if not (proc and proc.poll() is None): # Guard: Check handle EXISTS AND Process RUNNING..
            logging.debug("Stop server command ignored - process not running or handle lost.")
            self.server_process = None # Ensure handle cleared if checked confirmed not running.
            return True # Stop Intention met if Not RUNNING already.

        pid = proc.pid; port = self.last_known_port
        logging.info(f"Attempting STOP server (PID={pid}, Port={port})...")
        stopped = False
        try:
            proc.terminate(); proc.wait(timeout=3.5) # TERM + 3.5 sec wait ok
            if proc.poll() is None: raise subprocess.TimeoutExpired(proc.args, 3.5)
            logging.info(f"Server process (PID={pid}) terminated OK.")
            stopped = True
        except subprocess.TimeoutExpired:
            logging.warning(f"Server (PID={pid}) Terminate TIMEOUT -> Force KILL.")
            proc.kill(); proc.wait(timeout=5)
            if proc.poll() is not None: stopped=True ; logging.info(f"Server (PID={pid}) FORCE KILLED OK.");
            else: stopped=False; logging.error(f"SERVER PID={pid} process May PERSIST after KILL! manual Check Needed?") # KILL failed? Very Bad scenario..
        except Exception as e:
              logging.exception(f"Error during server stop procedure: {e}")
              stopped=False # ERR occurred.. assume stop NOT confirmed OK..
        finally:
             self.server_process = None # ALWAYS clear handle After attempt.
             self.last_known_port= None # Clear PORT number too.. ok.
             logging.info(f"Server stop sequence Complete. Final State Assumed = {'Stopped' if stopped else 'UNCERTAIN / ERROR stop'}")

        return stopped # Return the Result of stop attempt..

    def is_running(self) -> bool:
         """Checks if the server process handle exists and process is currently running."""
         return self.server_process is not None and self.server_process.poll() is None

    def check_status(self) -> Tuple[bool, str]:
        """Polls status, returns (isRunning, statusMessage), cleans up handle if stopped."""
        proc = self.server_process
        is_alive = False
        status_msg = "Server is INACTIVE."

        if proc and proc.poll() is None: # Handle exists and running fine
            is_alive = True
            status_msg = f"Server RUNNING (Port: {self.last_known_port})"
        elif proc : # Handle exists BUT proc died? ==> CRASH or Normal stop happened Outside this manager?
              rc = proc.returncode
              stderr_h = proc.stderr.read(150) if proc.stderr and not proc.stderr.closed else '(stderr closed?)' # Read snippet better log..
              logging.warning(f"Server process PID={proc.pid} Found STOPPED (Exit Code: {rc}). Stderr Hint = '{stderr_h}...' ")
              status_msg = f"Server STOPPED (rc={rc})"
              # >>>> CLEANUP the Internal HANDLE HERE <<<< Because process confirmed DEAD...
              self.server_process = None; self.last_known_port = None

        # Return final status DETECTED now.
        return (is_alive, status_msg)