from fastapi import FastAPI, Request, HTTPException, Response
from pydantic import BaseModel, Field
import requests
import json
import os
import yaml
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Union

# Setup basic logging for the proxy server
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ProxyServer - %(message)s')

# Import core components (handle potential direct run vs package import)
try:
    from .memory_graph import MemoryGraph
    from .retriever import ReflectiveRetriever
    from .memory_node import MemoryNode
except ImportError:
    logging.warning("Relative imports failed, likely running directly. Trying absolute.")
    # This might fail if not run from the project root or package not installed
    from memory_graph import MemoryGraph
    from retriever import ReflectiveRetriever
    from memory_node import MemoryNode

# --- Configuration and File Paths ---
# Assume files are relative to this script's location if run directly,
# or relative to the package installation otherwise.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "proxy_config.json")
MEMORY_GRAPH_PATH = os.path.join(SCRIPT_DIR, "memory_graph.json")
PROMPTS_FILE_PATH = os.path.join(SCRIPT_DIR, "prompts.yaml")

# --- Load Config ---
config = {}
BACKEND_URL = "http://localhost:11434/v1/chat/completions" # Fallback
MODEL_NAME = "mistral" # Fallback
try:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f: config = json.load(f)
    BACKEND_URL = config.get("backend_url") # URL *proxy* talks to
    MODEL_NAME = config.get("model_name")   # Model *proxy* uses with backend
    if not BACKEND_URL or not MODEL_NAME: raise ValueError("backend_url or model_name missing in config")
    logging.info(f"Proxy Config OK: Backend Target='{BACKEND_URL}', Model='{MODEL_NAME}'")
except Exception as e:
    logging.error(f"FATAL Proxy Error loading config '{CONFIG_PATH}': {e}. Using Fallbacks.", exc_info=False) # Don't need full trace for config load fail usually
    # Keep fallback values defined above

# --- Load Prompts ---
prompts = {}
DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful reasoning assistant. Answer based only on the provided context."
try:
    with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f: prompts = yaml.safe_load(f)
    if not isinstance(prompts, dict) or 'chat_system_prompt' not in prompts:
        logging.warning(f"'{PROMPTS_FILE_PATH}' missing 'chat_system_prompt'. Using default.")
    else:
        logging.info(f"Proxy Server: Loaded system prompt from {PROMPTS_FILE_PATH}")
except FileNotFoundError:
    logging.warning(f"Prompts file '{PROMPTS_FILE_PATH}' not found. Using default system prompt.")
except yaml.YAMLError as e:
    logging.warning(f"Failed parsing prompts YAML '{PROMPTS_FILE_PATH}': {e}. Using default system prompt.")
except Exception as e:
    logging.warning(f"Failed loading prompts from '{PROMPTS_FILE_PATH}': {e}. Using default system prompt.")

# --- Memory Components ---
memory_graph = MemoryGraph()
try:
    # Ensure graph file path exists before trying load
    if os.path.exists(MEMORY_GRAPH_PATH):
        memory_graph.load_from_json(MEMORY_GRAPH_PATH)
        logging.info(f"Proxy Server: Loaded {len(memory_graph.nodes)} nodes from {MEMORY_GRAPH_PATH}")
    else:
         logging.warning(f"Proxy Server: Memory graph file '{MEMORY_GRAPH_PATH}' not found. Starting empty.")
except Exception as e:
     logging.error(f"Proxy Server: Error loading memory graph '{MEMORY_GRAPH_PATH}': {e}. Starting empty.", exc_info=True) # Log full trace on graph load error

retriever = ReflectiveRetriever(memory_graph)

app = FastAPI(title="Lucid Memory Proxy Server (Ollama Compatible)")

# --- Ollama Compatible Request/Response Models ---
# (Keep these definitions exactly as before)
class OllamaChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = False
    template: Optional[str] = None
    keep_alive: Optional[Union[str, float]] = None

class OllamaChatResponseChoice(BaseModel):
    index: int = 0
    message: OllamaChatMessage
    finish_reason: str = "stop"

class OllamaChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-lucid-{os.urandom(10).hex()}") # Custom ID prefix
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    model: str # Model name *this proxy used*
    choices: List[OllamaChatResponseChoice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0} # Placeholder


# --- API Endpoint ---
@app.post(
    "/v1/chat/completions",
    response_model=OllamaChatCompletion,
    response_model_exclude_unset=True
)
async def ollama_compatible_chat(request: OllamaChatRequest):
    """
    Handles Ollama-compatible chat requests, injects context, calls backend, returns formatted response.
    """
    if not request.messages: raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    if request.stream: raise HTTPException(status_code=501, detail="Streaming responses not implemented by this proxy.")

    user_message_content = ""
    if request.messages[-1].role == "user": user_message_content = request.messages[-1].content
    if not user_message_content: raise HTTPException(status_code=400, detail="Last message must be from 'user'.")

    logging.info(f"Received Ollama Chat Request | User Msg: '{user_message_content[:80]}...' | Client Model Req: '{request.model}'")

    # 1. Memory Retrieval
    try:
        candidates = retriever.retrieve_by_keyword(user_message_content)
        best_nodes = retriever.reflect_on_candidates(candidates, user_message_content)
        logging.info(f"Retrieved {len(best_nodes)} relevant nodes (from {len(candidates)} candidates).")
    except Exception as retrieve_err:
         logging.error(f"Memory retrieval failed: {retrieve_err}", exc_info=True)
         best_nodes = [] # Continue without context on retrieval error

    # 2. Build Enhanced Prompt
    memory_prompt_section = ""
    if best_nodes:
        memory_bits = ["Relevant Memories:"]
        for i, node in enumerate(best_nodes, 1):
             node_info = [f"--- Node {i} (ID: {node.id}) ---"]
             if node.sequence_index is not None: node_info.append(f"[Seq: {node.sequence_index}, Parent: {node.parent_identifier}]")
             node_info.append(f"Summary: {node.summary}")
             if node.key_concepts: node_info.append("Key Concepts:\n" + "".join([f"- {c}\n" for c in node.key_concepts]))
             if node.dependencies: node_info.append("Dependencies:\n" + "".join([f" -> {d}\n" for d in node.dependencies]))
             if node.produced_outputs: node_info.append("Outputs:\n" + "".join([f" <- {o}\n" for o in node.produced_outputs]))
             if node.tags: node_info.append(f"Tags: {', '.join(node.tags)}")
             memory_bits.append("\n".join(node_info))
        memory_prompt_section = "\n\n".join(memory_bits) + "\n\n---\n\n"
    else:
        memory_prompt_section = "(No relevant memories found for this query.)\n\n"

    final_user_prompt_for_backend = memory_prompt_section + f"Based ONLY on the memories provided, answer:\nQuestion: {user_message_content}"

    # 3. Prepare Request to Configured Backend
    system_prompt_content = prompts.get('chat_system_prompt', DEFAULT_CHAT_SYSTEM_PROMPT)
    request_temp = 0.2 # Default temp
    if request.options and isinstance(request.options.get("temperature"), (float, int)): request_temp = request.options["temperature"]

    backend_payload = {
        "model": MODEL_NAME, # Use proxy's configured model
        "messages": [
            {"role": "system", "content": system_prompt_content},
            {"role": "user", "content": final_user_prompt_for_backend}
        ],
        "temperature": request_temp,
        "stream": False
    }

    # --- Prepare Headers (with Auth if key exists) ---
    backend_api_key = config.get("api_key", "") # Get key from loaded config
    backend_headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    if backend_api_key:
        backend_headers['Authorization'] = f"Bearer {backend_api_key}"
        logging.info(f"Proxy: Adding Authorization header (Key len: {len(backend_api_key)}).")
    else:
        logging.info("Proxy: No API key in config, sending request without Authorization header.")

    logging.info(f"Proxy->Backend Request: Target='{BACKEND_URL}', Model='{MODEL_NAME}'")
    logging.info(f"Proxy: Sending Headers: { {k: (v if k != 'Authorization' else 'Bearer ***') for k,v in backend_headers.items()} }") # Mask key in log
    logging.debug(f"Proxy: Sending Payload: {json.dumps(backend_payload, indent=2)}") # Log full payload at debug

    # 4. Call Backend and Handle Response
    assistant_content = "Error: Failed to get response from backend LLM." # Default error
    try:
        backend_response = requests.post( BACKEND_URL, json=backend_payload, headers=backend_headers, timeout=120 )
        backend_response.raise_for_status() # Check HTTP status
        backend_data = backend_response.json()
        logging.info(f"Proxy<-Backend Response: Status={backend_response.status_code}")

        # --- Extract assistant reply (assuming OpenAI-like) ---
        try:
            choices = backend_data.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message")
                if isinstance(message, dict): assistant_content = message.get("content", assistant_content)
            logging.debug(f"Extracted assistant content (len {len(assistant_content)}).")
        except Exception as parse_err:
             logging.error(f"Error parsing content from BACKEND response: {parse_err}", exc_info=False)
             logging.debug(f"Backend raw data: {backend_data}")
             assistant_content = f"Error parsing backend response structure. Raw: {str(backend_data)[:200]}..."

    except requests.exceptions.Timeout:
        logging.error(f"Proxy->Backend Timeout connecting to {BACKEND_URL}")
        raise HTTPException(status_code=504, detail="Request to backend LLM timed out.")
    except requests.exceptions.RequestException as e:
        # Log details and raise appropriate HTTP error
        status = getattr(e.response, 'status_code', 502) # Default to 502 if no response obj
        resp_text = getattr(e.response, 'text', '')[:200] # Limit response text log
        err_detail = f"Proxy failed to communicate with backend LLM: {e.__class__.__name__} for url: {BACKEND_URL} | Status: {status} | Resp: {resp_text}"
        logging.error(err_detail)
        # Forward the status code if it's a client/server error from backend (e.g., 401, 400, 404, 500)
        forward_status = status if isinstance(status, int) and status >= 400 else 502
        raise HTTPException(status_code=forward_status, detail=err_detail) # Raise with specific code
    except Exception as e:
        logging.exception("Unexpected error during backend call/parsing")
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {e}")

    # 5. Reformat into Ollama response structure
    logging.info("Proxy: Reformatting response to Ollama standard...")
    ollama_response = OllamaChatCompletion(
        model=MODEL_NAME, # Report the model *this proxy used*
        choices=[ OllamaChatResponseChoice( message=OllamaChatMessage(role="assistant", content=assistant_content) ) ]
        # usage is placeholder
    )
    return ollama_response


# --- Root Endpoint ---
@app.get("/")
async def root():
    nodes_count = len(memory_graph.nodes) if memory_graph else 0
    return {
        "message": "Lucid Memory Proxy Server (Ollama Compatible)",
        "memory_nodes_loaded": nodes_count,
        "configured_backend_url": BACKEND_URL,
        "configured_backend_model": MODEL_NAME
    }

# --- Main Block (for direct run) ---
if __name__ == "__main__":
    import uvicorn
    run_port = 8000
    try:
        # Try reading port from config if available
        if config and config.get('local_proxy_port'):
             run_port = int(config['local_proxy_port'])
    except (TypeError, ValueError):
         logging.warning(f"Invalid 'local_proxy_port' in config, using default {run_port}")

    print(f"\nStarting Lucid Memory Proxy Server...")
    print(f" - Listening on: http://0.0.0.0:{run_port}")
    print(f" - Target Backend: {BACKEND_URL}")
    print(f" - Target Model: {MODEL_NAME}")
    print(f" - Memory Graph: {MEMORY_GRAPH_PATH} ({len(memory_graph.nodes)} nodes loaded)")
    print(f" - System Prompt: '{prompts.get('chat_system_prompt', DEFAULT_CHAT_SYSTEM_PROMPT)[:50]}...'")
    print("(Use Ctrl+C to stop)")

    module_name = os.path.basename(__file__).replace('.py', '')
    try:
        uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=run_port, reload=True, log_level="info")
    except Exception as e:
         logging.critical(f"Failed to start Uvicorn server: {e}", exc_info=True)