import streamlit as st
import requests
import json
import logging
import threading
from typing import TYPE_CHECKING, Dict, List, Any

# ... (Imports and logging setup) ...

# --- Separate Function for Thread ---
def _send_chat_request_thread(url: str, payload: Dict, session_state: st.session_state):
    """Sends HTTP request in background. Updates session state."""
    logging.info(f"Chat Thread: Sending request to {url}...")
    ai_reply = ""
    error_message = None
    # --- Set Thinking Flag at Thread Start ---
    # This is slightly problematic if thread fails to start,
    # but simpler than complex callback for now. Controller approach would be better.
    # Assume success path first.
    # session_state.is_assistant_thinking = True # <<< Let main thread set this before starting

    try:
        # ... (requests.post and response processing logic) ...
        response = requests.post(url, json=payload, timeout=90.0); response.raise_for_status(); data = response.json()
        logging.info(f"Chat Thread: Received response status {response.status_code}")
        try:
            ai_reply = data['choices'][0]['message']['content']
            logging.info(f"Chat Thread: Successfully extracted reply (len {len(ai_reply)}).")
        except (IndexError, KeyError, TypeError) as parse_e: logging.error(f"Chat Thread: PARSE ERR proxy reply: {data}", exc_info=True); error_message = "⚠️ Error: Couldn't parse reply format."
    except requests.exceptions.Timeout: error_message = f"⏳ TIMEOUT contacting Proxy ({url})"; logging.error(error_message)
    except requests.exceptions.ConnectionError: error_message = f"🔌 CONNECTION REFUSED ({url}). Server running?"; logging.error(error_message)
    except requests.exceptions.RequestException as e: error_message = f"❌ HTTP Error ({getattr(e.response, 'status_code', 'N/A')})"; logging.error(f"{error_message}: {e}")
    except json.JSONDecodeError as json_e: error_message = "❌ Invalid JSON reply!"; logging.error(f"ProxyRespDecodeERR -> {json_e}")
    except Exception as e: error_message = f"💥 Unexpected Chat Error"; logging.exception("Chat proc Unexp Err")
    finally:
        # --- Update Session State ---
        message_to_add = {"role": "assistant", "content": error_message if error_message else ai_reply}
        if 'chat_history' not in session_state: session_state.chat_history = []
        session_state.chat_history.append(message_to_add)
        session_state.is_assistant_thinking = False # Mark thinking as done AFTER response/error
        session_state.new_message_received = True # Flag for main thread
        logging.info("Chat Thread: Updated session_state. Set new_message_received flag.")


# --- Main Rendering Function ---
def render_chat_tab(controller: 'LucidController', session_state: st.session_state):
    """Renders the Chat Assistant tab UI and handles interactions."""
    st.header("💬 Chat Assistant")

    # --- Initialize state if missing ---
    if 'chat_history' not in session_state: session_state.chat_history = []
    if 'prompt_to_send' not in session_state: session_state.prompt_to_send = None
    if 'is_assistant_thinking' not in session_state: session_state.is_assistant_thinking = False
    if 'new_message_received' not in session_state: session_state.new_message_received = False

    # --- Display Chat History ---
    current_chat_history = list(session_state.chat_history)
    for message in current_chat_history:
        with st.chat_message(message["role"]):
             # Add thinking indicator based on state
             if message["role"] == "assistant" and session_state.is_assistant_thinking and message == current_chat_history[-1]:
                 st.write("🤔 Thinking...") # Show thinking if it's the last message and flag is set
             else:
                 st.write(message["content"]) # Otherwise show content

    # --- Process pending request ---
    prompt_to_process = session_state.prompt_to_send
    # Only process if NOT currently thinking
    if prompt_to_process and not session_state.is_assistant_thinking:
        logging.info(f"Processing stored prompt: '{prompt_to_process[:60]}...'")
        session_state.prompt_to_send = None # Clear the flag/prompt

        # --- Set Thinking State BEFORE starting thread ---
        session_state.is_assistant_thinking = True
        # --- REMOVE the Thinking message append here ---
        # session_state.chat_history.append({"role": "assistant", "content": "🤔 Thinking..."})
        # --- REMOVE the st.rerun() here ---
        # st.rerun()

        # === Code below now executes in the *same* run after prompt stored ===
        cfg = controller.get_config()
        proxy_port = cfg.get('local_proxy_port', 8000)
        proxy_url = f"http://localhost:{proxy_port}/v1/chat/completions"
        payload = { "model": cfg.get("model_name", "default"), "messages": [{"role": "user", "content": prompt_to_process}], "temperature": 0.2, "stream": False }
        timeout = 90.0

        logging.info(f"Chat Request Prepared: URL={proxy_url}, Model={payload['model']}")

        try:
            logging.info("Attempting to start chat request thread...")
            chat_thread = threading.Thread( target=_send_chat_request_thread, args=(proxy_url, payload, session_state), daemon=True, name="ChatRequestThread" )
            chat_thread.start()
            logging.info("Chat request thread started.")
            # >>> We now rely on the is_assistant_thinking flag for UI feedback <<<
        except Exception as thread_e:
             logging.exception("Failed to start chat request thread!")
             error_msg = f"Error starting background chat task: {thread_e}"
             # Append error directly to history if thread fails to start
             session_state.chat_history.append({"role": "assistant", "content": error_msg})
             session_state.is_assistant_thinking = False # Ensure thinking stops
             st.rerun() # Rerun to show the startup error

    # --- Check if a new message was received from background thread ---
    if session_state.get('new_message_received'): # Use .get for safety
        logging.info("Detected new_message_received flag. Triggering rerun.")
        session_state.new_message_received = False # Reset flag
        st.rerun() # Rerun to display the new message added by the thread

    # --- Chat Input Area ---
    server_running, _ = controller.check_http_server_status()
    processing_now = session_state.get('is_processing', False)
    assistant_thinking = session_state.is_assistant_thinking
    chat_disabled = not server_running or processing_now or assistant_thinking

    # --- Input action ---
    if prompt := st.chat_input(
            "Ask about loaded context...",
            key="chat_input_main_tab",
            disabled=chat_disabled, # Disable input while thinking
        ):
        logging.info(f"Chat Input Received: '{prompt[:60]}...'")
        session_state.prompt_to_send = prompt
        session_state.chat_history.append({"role": "user", "content": prompt})
        # Only rerun needed is here, to show user message and trigger processing block
        st.rerun()