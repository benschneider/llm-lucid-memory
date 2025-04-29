# streamlit_app.py

import streamlit as st
import os
import logging
import time
from typing import Dict # Needed for type hinting

# Import the refactored backend controller
try:
    from lucid_memory.controller import LucidController
    from lucid_memory.memory_node import MemoryNode # Needed for type hinting
except ImportError:
     st.error("Failed to import Lucid Memory modules. Make sure it's installed or runnning from the correct directory.")
     st.stop()

# --- Page Configuration (Basic) ---
st.set_page_config(
    page_title="Lucid Memory GUI",
    layout="wide", # Use wide layout
    initial_sidebar_state="expanded", # Keep sidebar open initially
)

# --- Logging Setup ---
# Configure logging (Streamlit doesn't capture root logs easily, console is fine for now)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Controller - %(message)s')


# --- Initialize Controller in Session State ---
# st.session_state acts like a global dict that persists across reruns
if 'controller' not in st.session_state:
    try:
        st.session_state.controller = LucidController()
        logging.info("Controller initialized and stored in session state.")
    except Exception as e:
        logging.error(f"CRITICAL: Failed Controller Initialization: {e}", exc_info=True)
        st.error(f"Fatal Error initializing application backend: {e}")
        st.stop() # Stop app execution if controller fails

CTL: LucidController = st.session_state.controller # Get controller instance

# --- UI Callbacks (Update Session Sstate for UI Refresh) ---
# These are simple versions; could pass more data or use queues for complex updates
def update_status_display(message: str):
    """Callback function for the controller to update status."""
    st.session_state.status_message = message
    # Streamlit reruns implicitly when session state changes,
    # so the status display below will update automatically.

def handle_processing_completion(graph_changed: bool):
    """Callback for when digestion finishes."""
    st.session_state.is_processing = False # Update processing flag
    st.session_state.graph_changed_flag = graph_changed if 'graph_changed_flag' not in st.session_state else st.session_state.graph_changed_flag or graph_changed
    # Status message usually handled by the final status callback from processor

# --- Set Controller Callbacks ---
CTL.status_update_callback = update_status_display
CTL.completion_callback = handle_processing_completion

# --- Initialize Session State Variables ---
if 'status_message' not in st.session_state:
    st.session_state.status_message = CTL.get_last_status()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # List of {"role": role, "content": content} dicts
if 'is_processing' not in st.session_state: # Track background digestion
    st.session_state.is_processing = CTL.is_processing()
if 'graph_changed_flag' not in st.session_state:
    st.session_state.graph_changed_flag = False # Track if graph changed since last display refresh


# --- Sidebar: Configuration & Controls ---
with st.sidebar:
    st.header("⚙️ Configuration")
    current_config = CTL.get_config() # Get latest config

    conf_backend_url = st.text_input("Backend LLM URL", value=current_config.get('backend_url', ''))
    conf_model_name = st.text_input("Model Name", value=current_config.get('model_name', ''))
    conf_proxy_port = st.number_input("Local Proxy Port", value=current_config.get('local_proxy_port', 8000), min_value=1025, max_value=65535)

    if st.button("Save Configuration", key="save_config"):
        new_config = {
            "backend_url": conf_backend_url,
            "model_name": conf_model_name,
            "local_proxy_port": conf_proxy_port
        }
        if CTL.update_config(new_config):
             st.success("Configuration saved!")
             st.rerun() # Rerun to reflect potential readiness change
        else:
             st.error("Failed to save configuration.")

    # --- File Loader ---
    st.header("📄 Load Context")
    uploaded_file = st.file_uploader(
        "Choose a file (.py, .md, .txt)",
        type=['py', 'md', 'txt'],
        # Disable upload if digestor not ready or already processing
        disabled=not CTL.is_digestor_ready() or st.session_state.is_processing
    )

    if uploaded_file is not None:
        if st.button(f"Digest {uploaded_file.name}", key="start_digest", disabled=st.session_state.is_processing):
             # Need to save uploaded file temporarily to pass path to controller
             temp_dir = "temp_uploads"
             os.makedirs(temp_dir, exist_ok=True)
             temp_file_path = os.path.join(temp_dir, uploaded_file.name)

             try:
                 with open(temp_file_path, "wb") as f:
                     f.write(uploaded_file.getvalue())
                 logging.info(f"File saved temporarily to {temp_file_path}")

                 st.session_state.is_processing = True # Set processing flag
                 st.session_state.graph_changed_flag = False # Reset change flag
                 # Start processing in background (controller handles thread)
                 CTL.start_digestion_for_file(temp_file_path)
                 st.rerun() # Rerun immediately to show progress/disable button
             except Exception as e:
                 st.error(f"Error saving or starting digestion: {e}")
                 logging.error(f"File handling error: {e}", exc_info=True)
             finally:
                  # Optional: Clean up temp file? Might be better done by processor completion?
                  # For now, leave it.
                  pass

    if not CTL.is_digestor_ready():
         st.warning("Digestor not ready. Check config.", icon="⚠️")
    if st.session_state.is_processing:
        st.info("Processing context file in the background...", icon="⏳")


    # --- Server Controls ---
    st.header("🖥️ Proxy Server")
    server_running = CTL.server_process is not None and CTL.server_process.poll() is None # Check if known handle is running

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Start Server", key="start_server", disabled=server_running):
            if CTL.start_server():
                st.info("Attempting to start server...")
                time.sleep(2.5) # Give server time to start before checking
                CTL.check_server_status() # Update status
                st.rerun()
            else:
                st.error("Failed to start server (Check logs).")
    with col2:
        if st.button("Stop Server", key="stop_server", disabled=not server_running):
            CTL.stop_server()
            st.info("Attempting to stop server...")
            time.sleep(1) # Give time for status update via polling
            CTL.check_server_status() # Update status
            st.rerun() # Rerun to update button state

    # --- Optional: Server status poll timer ---
    # if st.checkbox("Auto-check server status", value=False):
    #     st.write("(Will trigger reruns periodically)") # Inform user
    #     # Needs more complex logic using thread to avoid blocking/rerun loop issues

# --- Main Area: Title, Status, Chat, Memory ---
st.title("🧠 Lucid Memory Interface")

# Status Bar Display (always visible at top)
st.info(st.session_state.status_message, icon="ℹ️")

col_main_1, col_main_2 = st.columns([2, 1]) # Chat takes 2/3, Memory takes 1/3

# --- Chat Interface ---
with col_main_1:
    st.subheader("💬 Chat")
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
             st.markdown(message["content"])

    # Chat input (at the bottom)
    if prompt := st.chat_input("Ask a question about the loaded context...",
                            disabled=not (CTL.server_process is not None and CTL.server_process.poll() is None)): # Only enable if server running
        # Add user message to history and display
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
             st.markdown(prompt)

        # Send to backend (kept simple via 'requests' in GUI thread for now)
        # TODO: Abstract this interaction maybe via Controller for better testability/async?
        proxy_url = f"http://localhost:{CTL.config.get('local_proxy_port', 8000)}/chat"
        payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.2}

        try:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):                     # Show thinking indicator
                   response=requests.post(proxy_url, json=payload, timeout=65) # Timeout for chat
                   response.raise_for_status()
                   data=response.json()
                   reply="Error: Could not parse chat response."      # Default reply
                   if isinstance(data.get("choices"), list) and len(data["choices"]) > 0:
                        msg=data["choices"][0].get("message")
                        if isinstance(msg, dict): reply=msg.get("content", reply)
                   st.markdown(reply)                        # Stream the response (markdown supports streaming)
            st.session_state.chat_history.append({"role": "assistant", "content": reply}) # Add response to history

        # Handle specific errors more gracefully
        except requests.exceptions.ConnectionError:
            st.error(f"Connection Error: Could not reach Proxy Server at {proxy_url}. Is it running?")
        except requests.exceptions.Timeout:
            st.error("Error: Request to Proxy Server timed out.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with proxy: {e}")
            logging.warning(f"Chat proxy error: {e}", exc_info=False)
        except Exception as e:
            st.error(f"Error processing chat: {e}")
            logging.error(f"Chat processing error", exc_info=True)

# --- Memory Node Display ---
with col_main_2:
    st.subheader("ℹ️ Memory Nodes")

    # Button to force refresh memory display if needed
    # Not strictly required due to session state reruns, but can be useful
    # if st.button("Refresh Memory View"):
    #    st.session_state.graph_changed_flag = True # Ensure refresh happens
    #    st.rerun()

    with st.container(height=500, border=False): # Scrollable container
        nodes: Dict[str, MemoryNode] = CTL.get_memory_nodes() # Get nodes from controller
        if not nodes:
             st.write("(Memory graph is empty or not loaded)")
        else:
             # Sort for display
             # Use node ID as key for sorting temporarily
             sorted_nodes = sorted(nodes.items(), key=lambda item: (getattr(item[1], 'sequence_index', float('inf')), item[0]))

             if not sorted_nodes:
                st.write("(No nodes available for display)")
             else:
                for node_id, node in sorted_nodes:
                    with st.expander(f"ID: {node_id}", expanded=False):
                        # Build display string cleanly
                        details = []
                        seq = getattr(node, 'sequence_index', 'N/A')
                        parent = getattr(node, 'parent_identifier', None)
                        deps = getattr(node, 'dependencies', None)
                        outs = getattr(node, 'produced_outputs', None)
                        meta = getattr(node,'source_chunk_metadata', None)

                        details.append(f"**Sequence:** {seq}")
                        if parent: details.append(f"**Parent:** {parent}")
                        if meta and meta.get('identifier'): details.append(f"**Source ID:** `{meta.get('identifier')}`")

                        details.append(f"**Summary:** {node.summary}")
                        details.append(f"**Tags:** `{(', '.join(node.tags) if node.tags else 'None')}`")

                        concepts = getattr(node, 'key_concepts', [])
                        details.append(f"**Key Concepts ({len(concepts)}):**")
                        details.extend([f"- `{c}`" for c in concepts] if concepts else ["  _(None)_"])

                        # Only show dependency/output info if non-empty
                        if deps:
                             details.append(f"**Dependencies ({len(deps)}):**")
                             details.extend([f"- `{d}`" for d in deps])
                        if outs:
                             details.append(f"**Produced Outputs ({len(outs)}):**")
                             details.extend([f"- `{o}`" for o in outs])

                        st.markdown("\n".join(details), unsafe_allow_html=True)

# Clear graph changed flag after display potentially refreshed
st.session_state.graph_changed_flag = False