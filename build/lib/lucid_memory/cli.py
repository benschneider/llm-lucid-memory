import subprocess
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_streamlit():
    """
    Finds and runs the streamlit_app.py using the streamlit command.
    """
    # Try finding streamlit_app.py relative to this file first
    # Assumes cli.py is in lucid_memory/ and streamlit_app.py is one level up
    cli_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(cli_dir)
    streamlit_app_script = os.path.join(project_root, "streamlit_app.py")

    # Fallback: if not found, assume it's in the current working directory
    # (This helps if running directly from source checkout root)
    if not os.path.exists(streamlit_app_script):
         streamlit_app_script = "streamlit_app.py" # Rely on CWD

    if not os.path.exists(streamlit_app_script):
        logging.error(f"Error: Cannot find streamlit_app.py.")
        logging.error("Please run 'lucid-memory' from the project's root directory,")
        logging.error("or ensure lucid-memory is correctly installed.")
        sys.exit(1)

    cmd = ["streamlit", "run", streamlit_app_script]
    logging.info(f"Executing: {' '.join(cmd)}")

    try:
        # Using subprocess.run blocks until streamlit exits
        # Use start_new_session=True maybe if needed on some OS to separate from terminal
        process = subprocess.Popen(cmd)
        process.wait() # Wait for the streamlit process to finish
    except FileNotFoundError:
        logging.error("Error: 'streamlit' command not found.")
        logging.error("Please make sure Streamlit is installed ('pip install streamlit')")
        logging.error("and that the 'streamlit' command is in your system's PATH.")
        sys.exit(1)
    except Exception as e:
       logging.error(f"An error occurred while running Streamlit: {e}", exc_info=True)
       sys.exit(1)

if __name__ == '__main__':
    # Allows running python -m lucid_memory.cli directly if needed
    run_streamlit()