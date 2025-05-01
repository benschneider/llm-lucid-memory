import streamlit as st
import os
import logging
from typing import Dict, List, Optional, Tuple

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - StreamlitApp(Main) - %(message)s')

# --- Backend Controller & UI Imports ---
try:
    from lucid_memory.controller import LucidController
    # Import UI Rendering Functions
    from ui.sidebar import render_sidebar
    from ui.chat_tab import render_chat_tab
    from ui.memory_tab import render_memory_tab
    from ui.system_tab import render_system_tab
except ImportError as import_err:
     st.error(f"Fatal Import Error ({import_err}). Check install/PYTHONPATH.")
     logging.exception(f"Core app import failed: {import_err}")
     st.stop()

# --- Initialize or Get Controller ---
if 'controller' not in st.session_state:
    try: st.session_state.controller = LucidController(); logging.info("Controller initialized.")
    except Exception as init_error: logging.exception(f"Controller init FAILED: {init_error}"); st.error(f"Backend Init Error: {init_error}"); st.stop()

CTL: LucidController = st.session_state.controller

# --- Define and Register UI Callbacks ---
def ui_update_status(message: str) -> None:
    """Callback from CONTROLLER. Updates MAIN status message session state ONLY."""
    # This callback NO LONGER needs to handle progress args
    st.session_state.status_message = message

def ui_handle_processing_completion(graph_changed: bool) -> None:
    """Callback when background task FINISHES. Resets UI state."""
    # Reset processing flag ONLY
    st.session_state.is_processing = False
    st.session_state.graph_last_change_status = graph_changed # Keep track if graph changed
    logging.info(f"Callback: BACKGROUND JOB Finished. Graph Changed = {graph_changed}")
    st.toast("Processing Complete!", icon="✅")
    # Always rerun to remove progress bar / re-enable controls
    st.rerun()

# --- Link callbacks TO the CONTROLLER Instance ---
CTL.status_update_callback = ui_update_status
CTL.completion_callback = ui_handle_processing_completion

# --- Initialize Session State Variables Needed by UI Renderers ---
# Fewer state vars needed now for progress
if 'status_message' not in st.session_state: st.session_state.status_message = CTL.get_last_status()
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'is_processing' not in st.session_state: st.session_state.is_processing = CTL.is_processing_active()
if 'graph_last_change_status' not in st.session_state: st.session_state.graph_last_change_status = False
if 'selected_api_type' not in st.session_state: st.session_state.selected_api_type = CTL.get_config().get('api_type', 'Ollama')
if 'confirm_clear_memory' not in st.session_state: st.session_state.confirm_clear_memory = False
# REMOVED: progress_value, progress_detail from session state


# ================================================
# ============= Main App Execution ===============
# ================================================
render_sidebar(CTL, st.session_state) # Render sidebar
st.title("Lucid Memory v0.3") # Main title
st.info(f"Status: {st.session_state.status_message}", icon="ℹ️") # Global status

# --- >>> UPDATED: Progress Display Section <<< ---
# --- Reads progress DIRECTLY from Controller attributes via get_progress() ---
if st.session_state.get('is_processing'): # Check processing flag in session state
    with st.container(border=True):
        col_prog_bar, col_refresh = st.columns([4, 1])

        with col_prog_bar:
             # ---> Get progress values FROM CONTROLLER <---
             current_step, total_steps, detail_msg = CTL.get_progress()
             # -------------------------------------------
             prog_val = 0.0
             if total_steps > 0: prog_val = min(1.0, float(current_step / total_steps))

             pct_str = f"{prog_val:.0%}"
             progress_bar_text = f"{detail_msg} [{pct_str}]" if detail_msg else pct_str
             try: st.progress(prog_val, text=progress_bar_text)
             except TypeError: st.progress(prog_val); st.caption(f"{detail_msg} ({pct_str})")

        with col_refresh:
             # Manual Refresh Button still useful to force redraw
             if st.button("🔄 Update View", key="force_progress_update_btn", help="Manually refresh UI to see latest processing status"):
                  st.rerun() # Force re-execution -> re-calls CTL.get_progress()

    st.divider()
# --- End UPDATED Progress Display Section ---


# --- Setup and Render Main Content TABS ---
tab_titles = ["💬 Chat", "🧠 Memory Nodes", "📊 System"]
tab_chat, tab_memory, tab_system = st.tabs(tab_titles)

with tab_chat:   render_chat_tab(CTL, st.session_state)
with tab_memory: render_memory_tab(CTL, st.session_state)
with tab_system: render_system_tab(CTL, st.session_state)

# --- App Footer ---
st.divider()
st.caption("Lucid Memory - Core Alpha")