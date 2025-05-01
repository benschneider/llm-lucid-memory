import streamlit as st
import os
import logging
import time
from typing import Dict, Any

# --- Type Hinting & Constants ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lucid_memory.controller import LucidController
    from lucid_memory.managers.config_manager import API_PRESETS

TEMP_UPLOAD_DIR = "temp_uploads"

def render_sidebar(controller: 'LucidController', session_state: st.session_state):
    """Renders the complete sidebar UI: Config, File Processing, Server Controls."""
    with st.sidebar:
        # ================================================
        # 1. Configuration Section (using st.form)
        # ================================================
        st.header("⚙️ Configuration")
        current_config = controller.get_config()
        api_presets = controller.get_api_presets()

        with st.form(key="config_form"):
            st.caption("Select API type and enter details. URLs usually point to the `/chat/completions` endpoint.")

            # --- API Type Selection ---
            available_api_types = list(api_presets.keys())
            # Read selection from session state for persistence across reruns within form interaction
            selected_api_type = session_state.get('selected_api_type', current_config.get('api_type', 'Ollama'))
            if selected_api_type not in available_api_types: selected_api_type = available_api_types[0] # Default check

            selected_api_type_widget = st.selectbox(
                "API Type:", options=available_api_types,
                index=available_api_types.index(selected_api_type),
                key="sb_api_type_select" # Use within-form key
            )

            # --- Determine Defaults Based on Selection ---
            preset_for_selected = api_presets.get(selected_api_type_widget, {})
            api_key_needed = preset_for_selected.get("needs_key", False)
            # Default URL from preset IF type changed, else use current saved config
            backend_url_value = preset_for_selected.get("url", "") if selected_api_type_widget != selected_api_type else current_config.get('backend_url', '')
            # Default API key only from saved config (clear if type changes and needed)
            api_key_value = "" if selected_api_type_widget != selected_api_type and api_key_needed else current_config.get('api_key', '')


            # --- Input Fields ---
            conf_backend_url = st.text_input("API Backend URL:", value=backend_url_value, key="ft_conf_url")

            conf_api_key = ""
            if api_key_needed:
                 conf_api_key = st.text_input("🔑 API Key:", type="password", value=api_key_value, key="ft_api_key")

            # --- Model Selection ---
            available_models = controller.get_available_models()
            current_model_name = current_config.get('model_name', '')
            display_models_options = available_models.copy()
            if current_model_name and current_model_name not in display_models_options:
                 display_models_options.insert(0, current_model_name)
            if not display_models_options: display_models_options = ["(No Models Available)"]

            default_idx = 0
            try: # Safe Index lookup..
                 if current_model_name in display_models_options: default_idx = display_models_options.index(current_model_name)
            except ValueError: pass

            conf_model_name = st.selectbox("Active LLM Model:", options=display_models_options, index=default_idx, key="ft_model_select", disabled=len(available_models)<1 and not current_model_name) # Disable better check empty and NO saved selection..ok..

            # --- Port Selection ---
            conf_proxy_port = st.number_input("Local Proxy Port:", value=current_config.get('local_proxy_port', 8000), min_value=1025, max_value=65535, key="ft_port_select")

            # --- Form Submit ACTION Button ----
            form_submitted = st.form_submit_button("💾 Save & Apply Config")

            # --- Handle Form SUBMIT --- (Only runs block IF submit button clicked..)
            if form_submitted:
                # Read state FROM Widgets *inside* FORM scope -> ENSURE accurate values.. ok.
                final_api_type = selected_api_type_widget # Use the selected TYPE from 'selectbox' Not session state var.. OK.
                # --> Get values entered / selected just before Submit ---> SAFE.
                config_to_save = {
                    "api_type": final_api_type,
                    "backend_url": conf_backend_url,
                    "model_name": conf_model_name if conf_model_name != "(No Models Available)" else "",
                    "local_proxy_port": int(conf_proxy_port),
                    "api_key": conf_api_key if api_key_needed else "" # Only save key if relevant INPUT was visible ok logic..
                }
                logging.info(f"Applying Configuration Save...") # Remove Dict Log maybe cleaner no need.. OK..
                with st.spinner("Saving..."):
                    if controller.update_config(config_to_save):
                        session_state.selected_api_type = final_api_type # <<<---- UpdateSESSION State *AFTER* successful Save/Apply completes --->>> OK Pattern>..!
                        st.success("✅ Configuration Saved!")
                        st.rerun() # Force refresh UI Reflect -> Component Status reloads.. etc Models List refreshed..
                    else:
                        st.error("❌ Config Save FAILED. Check Logs.")
        # === End of st.form() ===

        # --- Model Refresh Button (Separate Action -> Place OUTSIDE Form context flow ok..) ---
        if st.button("🔄 Refresh Model List", key="refresh_mdl_btn_sidebar", help="Re-query backend for available models."):
             with st.spinner("Asking backend for models..."):
                  controller.refresh_models_list()
             st.rerun() # Redraw sidebar show NEW models discovered ok..

        st.divider() # Separate Config Section.. visually..

        # ================================================
        # 2. File Processing Section
        # ================================================
        st.header("📄 Load & Process Document")

        # --- Component Readiness Info --- READ ONLY state display.. ok..
        can_process = controller.is_digestor_ready # Check MAIN requirement..
        embedder_stat = "✅ Ready" if controller.is_embedder_ready else ("⚠️ Config OK/API?" if controller.component_mgr.embedder else "❌ Fail/Off")
        st.caption(f"Digestor: {'✅ Ready' if can_process else '❌ Off / Config Fail'}")
        st.caption(f"Embedder: {embedder_stat}")

        # --- UI BUSY state FLAG -> Read From SESSION STATE (Updated Background Callback) --- YES correct.
        processing_busy = session_state.get('is_processing', False)

        # --- File Uploader ---
        uploaded_file = st.file_uploader(
            "Upload / Drag Document (.py, .md, .txt)", type=['py', 'md', 'txt'],
            key="file_uploader_main_sidebar",
            disabled=not can_process or processing_busy, # LOGIC -> Disabled if CANNOT Process OR If ALREADY BUSY processing! OK looks Correct.
            help="Upload a file. Digestor must be Ready."
        )

        # --- Process Button (Only SHOWN if file SELECTED..) ---
        if uploaded_file is not None:
            # Button logic -> Click triggers background start.. ok..
            if st.button(f"🚀 Process '{uploaded_file.name}'", key="start_process_btn_sidebar", disabled=processing_busy): # Disable button ONLY when Busy FLAG is True.. OK logic.
                try:
                    # Save the file locally first (Controller Thread needs PATH..)
                    os.makedirs(TEMP_UPLOAD_DIR, exist_ok=True)
                    full_temp_path = os.path.join(TEMP_UPLOAD_DIR, uploaded_file.name)
                    with open(full_temp_path, "wb") as f: f.write(uploaded_file.getvalue())
                    logging.info(f"Temp file created at: '{full_temp_path}' for processing.")

                    # --- SET Busy STATUS / Reset Progress Before START Async Task---
                    session_state.is_processing = True ## Flag ON -> UI should react...
                    session_state.progress_value = 0.0 # Reset
                    session_state.progress_detail = "Starting..." # Initial Msg..
                    session_state.graph_last_change_status = False # Reset..
                    # --- TRIGGER Backend Controller ---
                    controller.start_processing_pipeline(full_temp_path)
                    # -------------------------------
                    st.info("⏳ Background Job Started...") # User Feedback -> action REGISTERED.. ok..
                    st.rerun() # Force UI re-draw -> Show 'Processing..' state / disabled buttons + progress section etc.. YES ok.. RERUN NEEDED HERE..

                except Exception as e:
                     st.error(f"❗️ Error during File Save/Pre-process Step: {e}"); logging.exception("File Prep Error before Processor Start!")
                     session_state.is_processing = False # CRITCIAL --> Reset BUSY state IF PREP step errors itself! PREVENTS stuck UI.. NICE Catch logic required here!

        # --- PROGRESS Display Area --- (Conditional Visibility based on STATE FLAG...)
        if processing_busy: # Check the Session State Flag.. Correct.
            st.divider()
            st.subheader("📊 Processing...")
            current_progress_val = session_state.get('progress_value', 0.0) # Read Progress safely.. ok.
            current_detail_txt = session_state.get('progress_detail', "")
            pct_display_text = f"{current_progress_val:.0%}" # Nice Percent format string.. ok..
            # >> Use Try -> Fallback Display logic Robust OK >>
            try: st.progress(current_progress_val, text=f"{current_detail_txt} [{pct_display_text}]") # Show detail ON Bar if Version ok..
            except TypeError: st.progress(current_progress_val); st.caption(f"{current_detail_txt} ({pct_display_text})") # Else separate below ok fallback..
            st.divider()
        # ------------------------------


        # ================================================
        # 3. Proxy Server Controls
        # ================================================
        st.header("🖥️ Proxy Server")
        is_server_running, _ = controller.check_http_server_status() # Get Server status bool only ok.. POLL state each cycle..

        col_start, col_stop = st.columns(2)
        with col_start:
             # Start Button logic ok -> Disable IF Running.
            if st.button("🟢 Start", key="start_srv_btn_sidebar", disabled=is_server_running):
                 with st.spinner("Starting..."):
                      start_ok = controller.start_http_server() # Call the METHOD.. ok..
                      time.sleep(1) # Short PAUSE allow maybe backend process start up a bit more.. OK..
                      _, final_msg = controller.check_http_server_status() # Trigger FINAL Status Update after maybe started.. ok..
                      if start_ok : st.toast(f"Start Request -> OK ({final_msg})", icon="🚀")
                      else: st.error("❌ Start FAILED. See logs")
                 st.rerun() # Force UI update button state ok needed..

        with col_stop:
             # Stop Button logic ok -> Disable IF (NOT Running).. Correct..
            if st.button("🔴 Stop", key="stop_srv_btn_sidebar", disabled=not is_server_running):
                 with st.spinner("Stopping..."):
                     controller.stop_http_server() # Call the STOP method.. ok..
                     time.sleep(0.5); _, final_msg = controller.check_http_server_status(); # Short Delay + Final Status Check ok..
                     st.toast(f"Stop Request -> Sent ({final_msg})", icon="🛑")
                 st.rerun() # Refresh UI state show changes.. ok..

# --- End of render_sidebar ---