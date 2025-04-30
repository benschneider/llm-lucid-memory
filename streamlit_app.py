import streamlit as st
import os
import logging
import time
import json
import requests
from typing import Dict, List, Tuple # Added Tuple

# --- Page Configuration ---
st.set_page_config(
    page_title="Lucid Memory",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={'About': "Lucid Memory - Modular Reasoning Graph UI"}
)

# --- Logging Conf ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - StreamlitApp - %(message)s')

# --- Backend Controller Initialization & Handling ---
try:
    from lucid_memory.controller import LucidController, API_PRESETS
    from lucid_memory.memory_node import MemoryNode
except ImportError as import_err:
     st.error(f"Fatal Import Error ({import_err}). Check install/PYTHONPATH.")
     st.stop()

# --- Initialize or Get Controller from Session State ---
if 'controller' not in st.session_state:
    try:
        st.session_state.controller = LucidController()
        logging.info("LucidController initialized and stored.")
    except Exception as init_error:
        logging.exception(f"Controller initialization FAILED: {init_error}")
        st.error(f"Backend Initialization Error: {init_error}")
        st.stop()

CTL: LucidController = st.session_state.controller

# --- UI Callbacks (Link Controller signals to Streamlit state) ---
def ui_update_status(message: str):
    st.session_state.status_message = message

def ui_handle_processing_completion(graph_changed: bool):
    st.session_state.is_processing = False # Update flag
    st.session_state.graph_last_change_status = graph_changed # Track change
    logging.info(f"UI Notified: Processing Done. Graph Changed={graph_changed}")
    if graph_changed:
         st.rerun() # Force refresh ONLY if nodes were added/updated

# --- Register Callbacks ---
CTL.status_update_callback = ui_update_status
CTL.completion_callback = ui_handle_processing_completion

# --- Initialize Required Session State Variables ---
if 'status_message' not in st.session_state:
    st.session_state.status_message = CTL.get_last_status()
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = CTL.is_processing_active() # Use CURRENT controller state
if 'graph_last_change_status' not in st.session_state:
    st.session_state.graph_last_change_status = False
if 'selected_api_type' not in st.session_state: # Needed for API type change detection
    st.session_state.selected_api_type = CTL.get_config().get('api_type', list(API_PRESETS.keys())[0])

# ================================================
# =========== UI Rendering Functions =============
# ================================================

def render_sidebar(controller: LucidController):
    """Renders the complete sidebar UI."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        current_config = controller.get_config()

        # --- API Type Selection ---
        available_api_types = list(API_PRESETS.keys())
        selected_api_type = st.selectbox(
            "Select API Type:",
            options=available_api_types,
            index=available_api_types.index(st.session_state.selected_api_type) if st.session_state.selected_api_type in available_api_types else 0,
            key="api_type_select",
            help="Select backend type. Affects defaults and model fetching."
        )

        # --- Handle API Type Change & Determine Initial Values ---
        backend_url_value = current_config.get('backend_url', '')
        api_key_value = current_config.get('api_key', '')
        api_key_needed = API_PRESETS.get(selected_api_type, {}).get("needs_key", False)

        if st.session_state.selected_api_type != selected_api_type:
            logging.info(f"API Type changed to '{selected_api_type}'. Resetting URL/Key.")
            preset = API_PRESETS.get(selected_api_type)
            backend_url_value = preset.get("url", "") if preset else ""
            api_key_value = "" # Clear API key on type change
            # Force refresh models when API type/URL changes significantly? Maybe later
            st.session_state.selected_api_type = selected_api_type # Update state AFTER setting value

        # --- Input Fields ---
        conf_backend_url = st.text_input("API Backend URL (Chat Endpoint):", value=backend_url_value, key="conf_url_input")

        conf_api_key = ""
        if api_key_needed:
            conf_api_key = st.text_input("🔑 API Key:", type="password", value=api_key_value, key="conf_api_key_input")

        st.divider()

        # --- Model Selection ---
        col_model, col_refresh = st.columns([3, 1])
        with col_model:
            available_models = controller.get_available_models()
            current_model_name = current_config.get('model_name', '')
            display_options = available_models.copy()
            if current_model_name and current_model_name not in display_options:
                display_options.insert(0, current_model_name) # Ensure saved selection is shown

            default_idx = 0
            if display_options:
                 try: default_idx = display_options.index(current_model_name) if current_model_name in display_options else 0
                 except ValueError: default_idx = 0

            conf_model_name = st.selectbox(
                "Select Active Model:",
                options=display_options or ["(No models fetched)"], # Handle empty list
                index=default_idx,
                key="model_select_widget",
                disabled=not display_options # Disable if empty
            )
        with col_refresh:
            st.caption(" ") # Alignment hack
            if st.button("🔄 Models", key="refresh_models_btn", help="Refresh model list"):
                with st.spinner("Fetching models..."): controller.refresh_models_list()
                st.rerun()

        st.divider()

        # --- Proxy Port ---
        conf_proxy_port = st.number_input("Local Proxy Port", value=current_config.get('local_proxy_port', 8000), min_value=1025, max_value=65535, key="conf_port_input")

        # --- Save Button ---
        if st.button("💾 Save Configuration", key="save_config_btn_main"):
            new_config = {
                "api_type": selected_api_type,
                "backend_url": conf_backend_url,
                "model_name": conf_model_name if conf_model_name != "(No models fetched)" else "", # Save empty if nothing selected
                "local_proxy_port": int(conf_proxy_port),
                "api_key": conf_api_key if api_key_needed else ""
            }
            logging.info(f"Attempting Save Config: { {k: (v[:5]+'...' if k=='api_key' and v else v) for k,v in new_config.items()} }")
            with st.spinner("Saving & Reloading..."):
                if controller.update_config(new_config):
                    st.success("✅ Config Updated!")
                    time.sleep(0.5) # Short pause helpful?
                    st.rerun()
                else:
                    st.error("❌ Config Save FAILED.")

        # --- File Upload & Processing ---
        st.header("📄 Load & Process Document")
        digestor_ok = controller.is_digestor_ready
        embedder_ok = controller.is_embedder_ready # Can check this too
        can_process = digestor_ok # Main logic depends on Digestor primarily
        processing_busy = st.session_state.is_processing # Use flag updated by callbacks

        st.caption(f"Digestor Status: {'✅ Ready' if digestor_ok else '❌ Not Ready'}")
        st.caption(f"Embedder Status: {'✅ Ready' if embedder_ok else ('⚠️ Off' if controller.embedder else '❌ Fail')}")

        uploaded_file = st.file_uploader(
            "Upload Document (.py, .md, .txt)",
            type=['py', 'md', 'txt'],
            key="file_uploader_main",
            disabled=not can_process or processing_busy,
            help="Digestor must be Ready to process files."
        )

        if uploaded_file is not None:
            if st.button(f"🚀 Process '{uploaded_file.name}'", key="start_process_btn", disabled=processing_busy):
                temp_dir = "temp_uploads"
                try:
                    os.makedirs(temp_dir, exist_ok=True)
                    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(temp_file_path, "wb") as f: f.write(uploaded_file.getvalue())
                    logging.info(f"File ready at: '{temp_file_path}'")

                    st.session_state.is_processing = True
                    st.session_state.graph_last_change_status = False
                    controller.start_processing_pipeline(temp_file_path) # Start background job
                    st.info("⏳ Background processing initiated...")
                    st.rerun()
                except Exception as e:
                    st.error(f"❗️ File handling / Launch error: {e}")
                    logging.exception(f"File Upload/Start Processing Failed: {e}")

        if processing_busy:
            st.warning("⏳ Processing in background...", icon="🔄")

        # --- Proxy Server Controls ---
        st.header("🖥️ Proxy Server")
        srv_running, _ = controller.check_http_server_status()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🟢 Start", key="start_srv_btn", disabled=srv_running):
                with st.spinner("Starting Server..."):
                    if controller.start_http_server(): time.sleep(2); _, msg = controller.check_http_server_status(); st.toast(f"Start request sent ({msg})", icon ="🚀")
                    else: st.error("❌ Start FAILED");
                st.rerun()
        with col2:
            if st.button("🔴 Stop", key="stop_srv_btn", disabled=not srv_running):
                with st.spinner("Stopping Server..."):
                     controller.stop_http_server(); time.sleep(1); _, msg = controller.check_http_server_status(); st.toast(f"Stop request sent ({msg})", icon="🛑")
                st.rerun()


def render_chat_tab(controller: LucidController):
    """Renders the Chat Assistant tab."""
    st.header("💬 Chat Assistant")

    # Display History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
             st.markdown(message["content"], unsafe_allow_html=False)

    # Input processing
    server_running, _ = controller.check_http_server_status()
    prompt_disabled = not server_running or st.session_state.is_processing

    if prompt := st.chat_input("Ask a question...", key="chat_prompt_input", disabled=prompt_disabled):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
             st.markdown(prompt)

        # Send to backend proxy
        cfg = controller.get_config()
        proxy_url = f"http://localhost:{cfg.get('local_proxy_port', 8000)}/chat"
        payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        timeout = 90.0

        try:
            with st.chat_message("assistant"):
                spinner_msg = st.empty(); spinner_msg.markdown("🤔 Thinking...")
                response = requests.post(proxy_url, json=payload, timeout=timeout)
                spinner_msg.empty()
                response.raise_for_status()
                data = response.json()

                # Safer reply extraction
                reply = "Error: Failed to parse assistant reply."
                try:
                   reply = data['choices'][0]['message']['content']
                except (IndexError, KeyError, TypeError):
                   logging.error(f"Could not extract content from proxy response: {data}")

                st.markdown(reply)
            # Add response to history
            st.session_state.chat_history.append({"role": "assistant", "content": reply})

        except requests.exceptions.Timeout: st.error(f"TIMEOUT contacting Proxy ({timeout}s)")
        except requests.exceptions.ConnectionError: st.error(f"CONNECTION ERROR connecting to Proxy at {proxy_url}. Is it running?")
        except requests.exceptions.RequestException as e: st.error(f"HTTP Error ({e.response.status_code if e.response else 'N/A'}): {e}")
        except json.JSONDecodeError: st.error("Invalid JSON response from proxy."); logging.error(f"Proxy JSON Decode Err. RAW={response.text[:200]}")
        except Exception as e: st.error(f"Chat processing error: {e}"); logging.exception("Unexpected chat error")


def render_memory_tab(controller: LucidController):
    """Renders the Memory Node exploration tab."""
    nodes_dict: Dict[str, MemoryNode] = controller.get_memory_nodes()
    st.header(f"🧠 Explore Memory Nodes ({len(nodes_dict)} Loaded)")

    col1, col2 = st.columns([3, 1])
    with col1:
         # Sorting options
         sort_options = {
             "Sequence Index (Default)": lambda item: (getattr(item[1], 'sequence_index', float('inf')), item[0]),
             "Node ID (Alphabetical)": lambda item: item[0],
             "Parent ID": lambda item: (getattr(item[1], 'parent_identifier', ''), item[0])
         }
         selected_sort_key = st.selectbox("Sort Nodes By:", options=list(sort_options.keys()), index=0, key="node_sort_select")
    with col2:
         st.caption(" ") # Align
         if st.button("🔄 Refresh View", key="refresh_nodes_btn"): st.rerun()

    # Node Display Area
    with st.container(height=700): # Scrollable container
        if not nodes_dict:
             st.caption("(Memory graph is empty or not loaded.)")
        else:
            sorted_node_items = sorted(nodes_dict.items(), key=sort_options[selected_sort_key])
            if not sorted_node_items: st.caption("(No nodes to display)") # Should not happen if nodes_dict not empty?

            for node_id, node in sorted_node_items:
                 # Display each node in an expander
                 with st.expander(f"Node: `{node_id}`", expanded=False):
                     # Use helper function to build node details markdown? or keep here.. ok here for now..
                     details = []
                     # Metadata Line
                     seq = getattr(node, 'sequence_index', '-')
                     parent = getattr(node, 'parent_identifier', '-')
                     # src_meta = getattr(node, 'source_chunk_metadata', {}) # If we stored original meta.. skip now assume not..
                     # src_id = src_meta.get('identifier', '-') if src_meta else '-'
                     details.append(f"**Sequence:** `{seq}` | **Parent:** `{parent}`") # | **Source ID:** `{src_id}`")

                     # Summary
                     details.append(f"**Summary:**\n```\n{node.summary or '(None)'}\n```")

                     # Tags
                     tags_str = f"`{', '.join(node.tags)}`" if node.tags else "_(None)_"
                     details.append(f"**Tags:** {tags_str}")

                     # Key Concepts
                     concepts = getattr(node, 'key_concepts', [])
                     details.append(f"**Key Concepts ({len(concepts)}):**")
                     if concepts: details.extend([f"- `{c}`" for c in concepts])
                     else: details.append("  _(None)_")

                     # Dependencies (if present)
                     deps = getattr(node, 'dependencies', [])
                     if deps:
                          details.append(f"**Dependencies ({len(deps)}):**")
                          details.extend([f"- `{d}`" for d in deps])

                     # Outputs (if present)
                     outs = getattr(node, 'produced_outputs', [])
                     if outs:
                         details.append(f"**Produced Outputs ({len(outs)}):**")
                         details.extend([f"- `{o}`" for o in outs])

                     # Embedding Status
                     has_emb = getattr(node, 'embedding', None) is not None
                     emb_dim = len(node.embedding) if has_emb else 0
                     emb_stat = f"✅ Yes ({emb_dim}d)" if has_emb else "❌ No/Off"
                     details.append(f"**Embedding:** {emb_stat}")

                     # Raw JSON toggle
                     if st.checkbox("Show Raw JSON", key=f"raw_json_chk_{node_id}", value=False):
                         try:
                             node_dict_disp = node.to_dict()
                             node_dict_disp.pop('embedding', None) # Don't show huge vector
                             st.json(node_dict_disp, expanded=False)
                         except Exception as json_e: st.error(f"JSON display error: {json_e}")

                     # Render joined details
                     st.markdown("\n\n".join(details), unsafe_allow_html=True)


def render_system_tab(controller: LucidController):
    """Renders the System Info tab."""
    st.header("📊 System Status & Information")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("⚙️ Components")
        st.metric("Digestor Ready?", "✅ Ready" if controller.is_digestor_ready else "❌ Failed/Off")
        st.metric("Embedder Ready?", "✅ Ready" if controller.is_embedder_ready else ("⚠️ Off" if controller.embedder else "❌ Failed"))
        st.metric("Processing Task?", "⏳ Active" if controller.is_processing_active() else "✅ Idle")
        srv_running, srv_msg = controller.check_http_server_status()
        st.metric("Proxy Server?", "🟢 Running" if srv_running else "🔴 Stopped")
        st.caption(srv_msg) # Show detailed text below metric

    with col2:
         st.subheader("📦 Detected Models")
         st.caption(f"({len(controller.get_available_models())} found via current settings)")
         if st.button("🔄 Refresh Models Now", key="refresh_models_tab_btn"):
              with st.spinner("Fetching..."): controller.refresh_models_list();
              st.rerun()
         # Display List of Models
         with st.container(height=250):
            models = controller.get_available_models()
            if not models: st.caption("(None detected or fetch failed)")
            else:
                 for m in models: st.code(m, language=None) # Simple list using code blocks

    st.divider()
    st.subheader("🔧 Runtime Configuration")
    st.caption("(Currently active application settings)")
    try:
        st.json(controller.get_config(), expanded=False)
    except Exception as json_e: st.error(f"Error displaying config: {json_e}")


# ================================================
# ============= Main App Execution ===============
# ================================================

# --- Render Sidebar ---
render_sidebar(CTL)

# --- Render Main Area Tabs ---
st.title("Lucid Memory") # Keep title concise outside tabs
st.info(f"Status: {st.session_state.status_message}", icon="ℹ️") # Global status

tab_titles = ["💬 Chat Assistant", "🧠 Explore Memory Nodes", "📊 System Status"]
tab_chat, tab_memory, tab_system = st.tabs(tab_titles)

with tab_chat:   render_chat_tab(CTL)
with tab_memory: render_memory_tab(CTL)
with tab_system: render_system_tab(CTL)

# --- Optional: Add footer ---
st.divider()
st.caption("Lucid Memory Interface - v.alpha")