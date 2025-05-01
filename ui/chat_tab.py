import streamlit as st
import requests
import json
import logging
from typing import TYPE_CHECKING, Dict, List, Any

if TYPE_CHECKING:
    from lucid_memory.controller import LucidController

# Assume logger is configured at app level or configure simple one
# logging.basicConfig(...)

def render_chat_tab(controller: 'LucidController', session_state: st.session_state):
    """Renders the Chat Assistant tab UI and handles interactions."""
    st.header("💬 Chat Assistant")

    # Display Chat History (reading from session_state managed by main app loop)
    chat_history: List[Dict[str, str]] = session_state.get('chat_history', [])
    for message in chat_history:
        with st.chat_message(message["role"]):
             st.markdown(message["content"], unsafe_allow_html=False) # Display content safely

    # Chat Input Area
    server_running, _ = controller.check_http_server_status() # Check server state
    processing_now = session_state.get('is_processing', False) # Chck background task state
    chat_disabled = not server_running or processing_now # Determine enable/disable

    if prompt := st.chat_input(
            "Ask about loaded context...",
            key="chat_input_main_tab", # Unique key
            disabled=chat_disabled, # Control based on state
        ):

        logging.info(f"User sent chat msg: '{prompt[:60]}...'")
        # Append user msg to STATE -> Triggers Auto Rerun usually? yes..
        session_state.chat_history.append({"role": "user", "content": prompt})
        # OPTIONAL -> Trigger RERUN here to SHOW user message INSTANTLY.. Feels faster..
        st.rerun() # ---> Test if needed.. If chat msg appearance is delayed remove/add..

        # -> Prepare THEN Send User Prompt to Backend --> Run After Rerun to ensure user sees prompt..!
        # Retrieve config needed for API call endpoint..
        cfg = controller.get_config()
        proxy_url = f"http://localhost:{cfg.get('local_proxy_port', 8000)}/chat"
        payload = {"messages": [{"role": "user", "content": prompt}], "temperature": 0.2}
        timeout = 90.0

        ai_reply = "" # Define default EMPTY state ok..
        error_message = None # Track Potential Error message..

        try:
            # Display "Thinking..." only DURING THE API CALL.. Better method..
            with st.chat_message("assistant"): # Use Context message type.. ok..
                thinking_placeholder = st.empty(); thinking_placeholder.markdown("🤔 Thinking...")
                response = requests.post(proxy_url, json=payload, timeout=timeout)
                thinking_placeholder.empty() # Clear message AFTER response..
                response.raise_for_status() # Check HTTP OK status..
                data = response.json()

                # Safely Extract content.. Assume SPECIFIC OpenAI like structure BACK from Proxy Server ok.. Need align if proxy differnt format..
                try:
                    ai_reply = data['choices'][0]['message']['content']
                    st.markdown(ai_reply) # RENDER the valid Response.
                except (IndexError, KeyError, TypeError):
                     logging.error(f"PARSE ERR reply structure INVALID from proxy: {data}")
                     error_message = "⚠️ Error: Couldn't understand reply format from server."
                     # Display Error message IN CHAT HISTORY Below instead of Raising Exception Directly..? -> Keep Flow OK..

        # -- Handle / Display various API error states to user.. ---
        except requests.exceptions.Timeout: error_message = f"⏳ TIMEOUT contacting Proxy at {proxy_url}"
        except requests.exceptions.ConnectionError: error_message = f"🔌 CONNECTION REFUSED at {proxy_url}. Is server running?"
        except requests.exceptions.RequestException as e: error_message = f"❌ HTTP Error ({getattr(e.response, 'status_code', 'N/A')}): {e}"
        except json.JSONDecodeError: error_message = "❌ Invalid JSON reply from Proxy!"; logging.error(f"ProxyRespDEcodeERR-> `{getattr(response,'text','NoResponse?')[:250]}`")# Log raw hint..
        except Exception as e: error_message = f"💥 Unexpected Chat Error: {e}"; logging.exception("Chat proc Unexp Err")


        # ---> APPEND Assistant Response OR ERRROR Msg TO HISTORY <---- After ALL try/catch logic completes ok..
        message_to_add = {"role": "assistant", "content": error_message if error_message else ai_reply}
        session_state.chat_history.append(message_to_add)

        # Force one final redraw Maybe Necessary IF AI response handling is complex? --> Maybe not Here test first...
        # st.rerun()