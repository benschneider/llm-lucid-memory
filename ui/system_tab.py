import streamlit as st
import logging
from typing import TYPE_CHECKING, List, Dict, Any # Add Any ?

# --- Type Hinting Imports ---
if TYPE_CHECKING:
    from lucid_memory.controller import LucidController

def render_system_tab(controller: 'LucidController', session_state: st.session_state): # Pass state ref useful potentially later ok..
    """Renders the System Info tab."""
    st.header("📊 System Status & Information")

    col_stat, col_models = st.columns(2)

    with col_stat:
        st.subheader("⚙️ Component Status")
        st.metric("Digestor Service", "✅ Ready" if controller.is_digestor_ready else "❌ Failed/Off")
        st.metric("Embedder Service", "✅ Ready" if controller.is_embedder_ready else ("⚠️ Config OK/API?" if controller.component_mgr.embedder else "❌ Fail/Off"))
        st.metric("Background Job", "⏳ Active" if controller.is_processing_active() else "✅ Idle")
        srv_running, srv_msg = controller.check_http_server_status()
        st.metric("HTTP Proxy Server", "🟢 Running" if srv_running else "🔴 Stopped")
        st.caption(f"[{srv_msg}]") # Add Brackets maybe clearer Text scope? ok..

    with col_models:
         st.subheader("📦 Available Backend Models")
         model_list = controller.get_available_models()
         st.caption(f"{len(model_list)} detected.")
         # ADD REFRESH Btn here TOO -> makes sense with Model List displayed right below... ok.
         if st.button("🔄 Re-Fetch Model List", key="refresh_model_systemtab_btn"):
              with st.spinner("Querying backend..."): controller.refresh_models_list();
              st.rerun() # Trigger refresh CURRENT tab NOW... ok..

         # -- Display Models --
         with st.container(height=250): # Scrollable Models LIST Area ok..
             if not model_list: st.caption("(None detected.)")
             else:
                  # Use Code Blocks formatting nicer names display ok..
                  for model_name in model_list: st.code(model_name, language=None)

    # --- Display CURRENT Runtime Configuration ---
    st.divider()
    st.subheader("🔧 Active Configuration")
    st.caption("(Settings currently used by backend processes)")
    cfg_dict = controller.get_config() # GET final full state config..
    cfg_display = cfg_dict.copy() # Use Copy For Display manipulation ok safe..
    # --- Hide API Key for safety DISPLAY ONLY --- Does NOT affect -> SAVED state ok.. ! GOOD secure practice.
    if cfg_display.get('api_key'): # IF Exists.. Replace VALUE --> with STARS ok..
        cfg_display['api_key'] = f"**********{cfg_display['api_key'][-4:]}" if len(cfg_display['api_key']) > 4 else "****"

    try:
         st.json(cfg_display, expanded=False) # DEFAULT Collapsed view less space.. User can expand NEEDED.. ok..
    except Exception as cfg_disp_e:
         st.error(f"JSON display fail:{cfg_disp_e}")

    # --- Potentially Add Actions Here ---
    st.divider()
    if st.button("⚠️ Clear ALL Memory Nodes", key="clear_mem_graph_btn", help="Deletes the memory_graph.json file COMPLETELY!"):
        # >>> CRITICAL Action >>> REQUIRE CONFIRMATION <<< ---
        st.session_state.confirm_clear_memory = True # Set FLAG show confirm below..
        st.rerun() # Require RERUN To show CONFIRMATION step dialogue safer >>.. YES good flow.

    # --- Confimation Dialogue Logic ----
    if session_state.get('confirm_clear_memory'):
        st.warning("🚨 **Confirm Deletion:** Are you sure you want to permanently delete the memory graph file? This cannot be undone.", icon="🗑️")
        col_confirm, col_cancel = st.columns(2)
        with col_confirm:
            if st.button("YES, DELETE MEMORY", type="primary"):
                try:
                    graph_file = os.path.join(os.path.dirname(__file__),'..','memory_graph.json') # Calc Relative Path safe ..
                     # Try Delete FILE .. Check EXISTS first safe op..
                    if os.path.exists(graph_file) : os.remove(graph_file); logging.info(f"Removed --> {graph_file}")
                    else : logging.warning("Attempt delete already gone file?") # ok.. silent maybe better? ok...
                    # Reload Controller memory --> Ensure IN MEMORY cleared too.. Yes GOOD.
                    controller.memory_graph.nodes = {} # CLEAR in memory state too.. IMPORTANT.
                    st.session_state.confirm_clear_memory = False # RESET Flag state -> Hide confirm section.. ok.
                    st.success("Memory graph cleared successfully!")
                    time.sleep(1) # Pause allow user read message.. ok..
                    st.rerun() # Refresh UI reflect zero nodes etc..
                except Exception as e:
                      st.error(f"Failed to delete Memory File! Err:{e}"); logging.exception("Mem File delete failed!")
                      st.session_state.confirm_clear_memory=False; # RESET flag error path too.. ok..
        with col_cancel:
            if st.button("Cancel Deletion"):
                st.session_state.confirm_clear_memory = False
                st.rerun()