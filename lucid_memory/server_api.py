import logging
import os
import json
import yaml
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Union

from fastapi import FastAPI, Request, HTTPException, APIRouter, Body
from pydantic import BaseModel, Field

# Core Lucid Memory components
from .controller import LucidController
from .memory_graph import MemoryGraph
from .retriever import ReflectiveRetriever
from .memory_node import MemoryNode
from .config_manager import ConfigManager
from .chunker import chunk as lib_chunk_content

# Setup basic logging
# Application-level logging configuration should ideally be in the runner script.
# For now, basic config here if run directly.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ServerAPI - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration and Global Instances ---
# The server instantiates its own ConfigManager and LucidController.
# This makes the server self-contained.
config_manager = ConfigManager()
server_config = config_manager.get_config()

# Paths - consider making these configurable via server_config if needed
LUCID_MEMORY_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MEMORY_GRAPH_PATH = os.path.join(LUCID_MEMORY_DIR, "memory_graph.json")
DEFAULT_PROMPTS_FILE_PATH = os.path.join(LUCID_MEMORY_DIR, "prompts.yaml")

MEMORY_GRAPH_PATH = server_config.get("memory_graph_path", DEFAULT_MEMORY_GRAPH_PATH)
PROMPTS_PATH = server_config.get("prompts_path", DEFAULT_PROMPTS_FILE_PATH)

# Initialize LucidController
# The controller will load/manage its own components based on the config
lucid_controller = LucidController(memory_graph_path=MEMORY_GRAPH_PATH)

# For the proxy part, we need a MemoryGraph instance (could be the one in controller)
# and a Retriever.
memory_graph_for_proxy: MemoryGraph = lucid_controller.memory_graph # Share the controller's graph
retriever_for_proxy = ReflectiveRetriever(memory_graph_for_proxy)

# Load prompts for the proxy
proxy_prompts = {}
DEFAULT_PROXY_SYSTEM_PROMPT = "You are a helpful reasoning assistant. Answer based only on the provided context."
try:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        proxy_prompts = yaml.safe_load(f)
    if not isinstance(proxy_prompts, dict) or 'chat_system_prompt' not in proxy_prompts:
        logger.warning(f"Prompts file '{PROMPTS_PATH}' missing 'chat_system_prompt' for proxy. Using default.")
except FileNotFoundError:
    logger.warning(f"Prompts file '{PROMPTS_PATH}' not found for proxy. Using default system prompt.")
except Exception as e:
    logger.warning(f"Failed loading prompts from '{PROMPTS_PATH}' for proxy: {e}. Using default system prompt.")

# FastAPI App Initialization
app = FastAPI(
    title="Lucid Memory Unified Server",
    description="Provides APIs for managing Lucid Memory knowledge graph and an LLM proxy.",
    version=server_config.get("version", "0.2.5")
)

# --- Pydantic Models for Library Control API ---
class ChunkRequest(BaseModel):
    source_code: str
    identifier: Optional[str] = "api_chunk_input.py"

class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]

class ProcessRequest(BaseModel):
    chunks: List[Dict[str, Any]]
    original_filename: str = "api_processed_file.dat"

class MemoryNodeResponse(BaseModel):
    memory_nodes: List[Dict[str, Any]] # List of MemoryNode.to_dict()

class AddNodeRequest(BaseModel):
    memory_node: Dict[str, Any]

class GenericSuccessResponse(BaseModel):
    success: bool
    message: Optional[str] = None

class UpdateConfigRequest(BaseModel):
    new_config_values: Dict[str, Any]

class StatusResponse(BaseModel):
    status: str
    processing_active: bool
    last_controller_status: str
    controller_config: Dict[str, Any]
    proxy_config: Dict[str, Any]
    memory_graph_nodes: int

# --- Library Control API Router ---
router_library_control = APIRouter(prefix="/library", tags=["Library Control"])

@router_library_control.post("/chunk", response_model=ChunkResponse)
async def chunk_route(request_data: ChunkRequest):
    logger.info(f"Received /library/chunk request for identifier: {request_data.identifier}")
    if not request_data.source_code.strip():
        raise HTTPException(status_code=400, detail="Source code cannot be empty.")
    try:
        # lib_chunk_content is an alias for lucid_memory.chunker.chunk
        chunks_result = lib_chunk_content(request_data.source_code, file_identifier=request_data.identifier)
        return ChunkResponse(chunks=chunks_result)
    except Exception as e:
        logger.error(f"Error in /library/chunk: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router_library_control.post("/process", response_model=MemoryNodeResponse)
async def process_route(request_data: ProcessRequest):
    logger.info(f"Received /library/process request for filename: {request_data.original_filename}, chunks: {len(request_data.chunks)}")
    if not request_data.chunks:
        raise HTTPException(status_code=400, detail="Chunks list cannot be empty.")
    try:
        # This is a synchronous (blocking) call within the FastAPI worker
        processed_nodes = lucid_controller.process_chunks_synchronously(
            request_data.chunks,
            request_data.original_filename
        )
        return MemoryNodeResponse(memory_nodes=[node.to_dict() for node in processed_nodes])
    except Exception as e:
        logger.error(f"Error in /library/process: {e}", exc_info=True)
        # Differentiate controller-specific errors (e.g., "processing active")
        if "processing task is already active" in str(e):
            raise HTTPException(status_code=409, detail=str(e)) # Conflict
        raise HTTPException(status_code=500, detail=str(e))

@router_library_control.post("/nodes/add", response_model=GenericSuccessResponse)
async def add_memory_node_route(request_data: AddNodeRequest):
    logger.info(f"Received /library/nodes/add request for node ID: {request_data.memory_node.get('id', 'UNKNOWN')}")
    if not request_data.memory_node:
        raise HTTPException(status_code=400, detail="Memory node data is required.")
    success = lucid_controller.add_memory_node_data(request_data.memory_node)
    if success:
        return GenericSuccessResponse(success=True, message="Memory node added successfully.")
    else:
        raise HTTPException(status_code=500, detail="Failed to add memory node.")

@router_library_control.get("/nodes", response_model=MemoryNodeResponse)
async def get_memory_nodes_route():
    logger.info("Received /library/nodes (GET) request.")
    memory_nodes_map = lucid_controller.get_all_memory_nodes()
    memory_nodes_list = [node.to_dict() for node in memory_nodes_map.values()]
    return MemoryNodeResponse(memory_nodes=memory_nodes_list)

@router_library_control.post("/config/update", response_model=GenericSuccessResponse)
async def update_config_route(request_data: UpdateConfigRequest):
    logger.info(f"Received /library/config/update request with keys: {list(request_data.new_config_values.keys())}")
    if not request_data.new_config_values:
        raise HTTPException(status_code=400, detail="New config values are required.")
    success = lucid_controller.update_config_values(request_data.new_config_values)
    if success:
        # Update server_config as well, as controller's config_mgr was updated
        global server_config
        server_config = lucid_controller.config_mgr.get_config() # Re-fetch
        return GenericSuccessResponse(success=True, message="Configuration updated successfully. Components reloaded.")
    else:
        # Controller/ConfigManager should log the specific reason for failure
        raise HTTPException(status_code=400, detail="Failed to update configuration. Check server logs for details (e.g., validation errors).")

@router_library_control.get("/status", response_model=StatusResponse)
async def library_status_route():
    logger.info("Received /library/status request.")
    controller_cfg = lucid_controller.get_config()
    proxy_cfg_summary = {
        "backend_url": server_config.get("backend_url"),
        "model_name": server_config.get("model_name"),
        "embedding_api_model_name": server_config.get("embedding_api_model_name"),
        "api_key_present_for_proxy": bool(server_config.get("api_key"))
    }
    return StatusResponse(
        status="Lucid Memory Library API is running.",
        processing_active=lucid_controller.is_processing_active(),
        last_controller_status=lucid_controller.get_last_status(),
        controller_config={
            "api_type": controller_cfg.get("api_type"),
            "model_name_for_digestor": controller_cfg.get("model_name"),
            "backend_url_for_digestor": controller_cfg.get("backend_url"),
            "embedding_model_for_embedder": controller_cfg.get("embedding_api_model_name"),
            "digestor_ready": lucid_controller.is_digestor_ready,
            "embedder_ready": lucid_controller.is_embedder_ready
        },
        proxy_config=proxy_cfg_summary,
        memory_graph_nodes=len(lucid_controller.memory_graph.nodes)
    )

app.include_router(router_library_control)

# --- LLM Proxy API Router (Ollama Compatible) ---
# Models for Ollama compatible proxy
class OllamaChatMessage(BaseModel):
    role: str
    content: str
    images: Optional[List[str]] = None

class OllamaChatRequest(BaseModel):
    model: str
    messages: List[OllamaChatMessage]
    format: Optional[str] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = Field(default=False)
    template: Optional[str] = None
    keep_alive: Optional[Union[str, float]] = None

class OllamaChatResponseChoice(BaseModel):
    index: int = 0
    message: OllamaChatMessage
    finish_reason: str = "stop"

class OllamaChatCompletion(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-lucidunified-{os.urandom(10).hex()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now(timezone.utc).timestamp()))
    model: str
    choices: List[OllamaChatResponseChoice]
    usage: Dict[str, int] = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

router_llm_proxy = APIRouter(prefix="/v1", tags=["LLM Proxy"])

@router_llm_proxy.post("/chat/completions", response_model=OllamaChatCompletion, response_model_exclude_unset=True)
async def ollama_compatible_chat_proxy(request: OllamaChatRequest):
    logger.info(f"Proxy (/v1/chat/completions) received request for model: {request.model}, messages: {len(request.messages)}")
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")
    if request.stream:
        # TODO: Implement streaming if needed in the future.
        # For now, return error or handle non-streamed and ignore stream flag.
        logger.warning("Streaming requested but not implemented by this proxy endpoint. Processing non-streamed.")
        # Fall through to non-streaming logic, client might handle it.
        # Or raise HTTPException(status_code=501, detail="Streaming responses not implemented by this proxy.")

    chat_history_for_retriever = []
    if request.messages:
        chat_history_for_retriever = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        user_message_content = chat_history_for_retriever[-1]["content"] # Ensure user_message_content is still the latest
    else:
        raise HTTPException(status_code=400, detail="Messages list cannot be empty.")

    try:
        # Step 1: Initial keyword-based retrieval
        keyword_candidates = retriever_for_proxy.retrieve_by_keyword(user_message_content)
        logger.info(f"Proxy: Initial keyword retrieval found {len(keyword_candidates)} candidates.")

        # Step 2: Reflect/Rank initial candidates (optional, but good to keep)
        # This helps select the best starting points for graph traversal.
        best_initial_nodes = retriever_for_proxy.reflect_on_candidates(keyword_candidates, user_message_content)
        logger.info(f"Proxy: Reflected on candidates, selected {len(best_initial_nodes)} best initial nodes.")

        # Step 3: Expand context using GraphRAG (if there are initial nodes)
        if best_initial_nodes:
            # Define max_neighbors for GraphRAG - this is the total nodes you want in the end
            # It should be less than or equal to the display limit later.
            max_graph_context_nodes = 7  # How many related nodes to pull in total
            final_context_nodes = retriever_for_proxy.retrieve_graph_context(
                initial_nodes=best_initial_nodes, # Pass the refined list
                query=user_message_content, # Current user query
                chat_history=chat_history_for_retriever,
                max_neighbors_to_include=max_graph_context_nodes
            )
            logger.info(f"Proxy: GraphRAG (with history awareness placeholder) expanded to {len(final_context_nodes)} contextual nodes.")
        else:
            final_context_nodes = []  # No initial nodes, so no graph context
            logger.info("Proxy: No initial nodes found, skipping GraphRAG expansion.")

        # `best_nodes` will now be `final_context_nodes` for prompt construction
        nodes_for_prompt = final_context_nodes

    except Exception as retrieve_err:
        logger.error(f"Proxy: Full retrieval pipeline (Keyword + GraphRAG) failed: {retrieve_err}", exc_info=True)
        nodes_for_prompt = []

    memory_prompt_section = ""
    if nodes_for_prompt:
        # You might want to adjust the limit here if retrieve_graph_context already limits
        # Or ensure the display limit here is consistent with what GraphRAG might return
        display_limit_in_prompt = 5  # How many of the final nodes to actually put in the prompt

        memory_bits = [f"Relevant Memories (max {display_limit_in_prompt} of {len(nodes_for_prompt)} shown, prioritized by relevance/structure):"]

        # The nodes_for_prompt might already be sorted by retrieve_graph_context.
        # If not, you might want a simple sort here, e.g., by sequence index if that makes sense for the context.
        for i, node in enumerate(nodes_for_prompt[:display_limit_in_prompt], 1):
            # ... (the rest of the node_info construction remains the same) ...
            node_info = [f"--- Memory {i} (ID: {node.id}) ---"]
            if node.sequence_index is not None:
                node_info.append(f"[Seq: {node.sequence_index}, Parent: {node.parent_identifier}]")
            node_info.append(f"Summary: {node.summary}")
            if node.key_concepts:
                node_info.append("Key Concepts/Logic:\n" + "".join([f"- {c}\n" for c in node.key_concepts]))
            if node.dependencies:
                node_info.append("Dependencies:\n" + "".join([f" -> {d}\n" for d in node.dependencies]))
            if node.produced_outputs:
                node_info.append("Outputs:\n" + "".join([f" <- {o}\n" for o in node.produced_outputs]))
            if node.tags:
                node_info.append(f"Tags: {', '.join(node.tags)}")
            memory_bits.append("\n".join(node_info))
        memory_prompt_section = "\n\n".join(memory_bits) + "\n\n---\n\n"
    else:
        memory_prompt_section = "(No relevant memories found for this query after retrieval.)\n\n"

    final_user_prompt_for_backend = memory_prompt_section + f"Based ONLY on the memories provided, answer the following question:\nQuestion: {user_message_content}"

    # Use server_config for backend LLM details
    backend_llm_url = server_config.get("backend_url")
    backend_llm_model = server_config.get("model_name")  # This is the model the proxy uses for its backend
    backend_api_key = server_config.get("api_key")

    if not backend_llm_url or not backend_llm_model:
        logger.error("Proxy: Backend LLM URL or model name not configured in server_config.")
        raise HTTPException(status_code=500, detail="Proxy backend not configured.")

    system_prompt = proxy_prompts.get('chat_system_prompt', DEFAULT_PROXY_SYSTEM_PROMPT)
    temperature = 0.2
    if request.options and isinstance(request.options.get("temperature"), (float, int)):
        temperature = request.options["temperature"]

    backend_payload = {
        "model": backend_llm_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": final_user_prompt_for_backend}
        ],
        "temperature": temperature,
        "stream": False
    }

    backend_headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
    if backend_api_key:
        backend_headers['Authorization'] = f"Bearer {backend_api_key}"

    logger.info(f"Proxy->Backend Request: Target='{backend_llm_url}', Model='{backend_llm_model}'")
    if logger.isEnabledFor(logging.DEBUG):  # Avoids string formatting if not debug
        logger.debug(f"Proxy->Backend Headers: { {k: (v if k != 'Authorization' else 'Bearer ***') for k,v in backend_headers.items()} }")
        logger.debug(f"Proxy->Backend Payload: {json.dumps(backend_payload, indent=2)}")

    assistant_content = "Error: Proxy failed to get response from backend LLM."
    try:
        import requests  # Keep import local to where it's used if it's heavy
        backend_response = requests.post(backend_llm_url, json=backend_payload, headers=backend_headers, timeout=120)
        backend_response.raise_for_status()
        backend_data = backend_response.json()
        logger.info(f"Proxy<-Backend Response: Status={backend_response.status_code}")

        choices = backend_data.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message")
            if isinstance(message, dict):
                assistant_content = message.get("content", assistant_content)
        logger.debug(f"Proxy: Extracted assistant content (len {len(assistant_content)}).")

    except requests.exceptions.Timeout:
        logger.error(f"Proxy->Backend Timeout to {backend_llm_url}")
        raise HTTPException(status_code=504, detail="Request to backend LLM timed out.")
    except requests.exceptions.RequestException as e:
        status = getattr(e.response, 'status_code', 502)
        resp_text = getattr(e.response, 'text', '')[:200]
        err_detail = f"Proxy->Backend communication error: {e.__class__.__name__} for {backend_llm_url} | Status: {status} | Resp: {resp_text}"
        logger.error(err_detail)
        raise HTTPException(status_code=status if status >= 400 else 502, detail=err_detail)
    except Exception as e:
        logger.exception("Proxy: Unexpected error during backend call/parsing")
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {str(e)}")

    return OllamaChatCompletion(
        model=backend_llm_model,
        choices=[OllamaChatResponseChoice(message=OllamaChatMessage(role="assistant", content=assistant_content))]
    )

app.include_router(router_llm_proxy)

# --- Root Endpoint ---
@app.get("/", tags=["Root"])
async def root():
    graph_last_modified = "N/A"
    if os.path.exists(MEMORY_GRAPH_PATH):
        try:
            graph_last_modified = datetime.fromtimestamp(os.path.getmtime(MEMORY_GRAPH_PATH)).isoformat()
        except Exception:
            pass

    server_port_to_display = server_config.get("unified_server_port", 8081)

    return {
        "message": "Welcome to Lucid Memory Unified Server API",
        "version": server_config.get("version", "0.2.5"),
        "documentation": ["/docs", "/redoc"],
        "library_api_status_url": "/library/status",
        "llm_proxy_chat_endpoint": "/v1/chat/completions",
        "memory_graph_nodes": len(memory_graph_for_proxy.nodes),
        "memory_graph_last_modified": graph_last_modified,
        "server_port": server_port_to_display
    }

# --- Cleanup on Shutdown ---
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Lucid Memory Unified Server shutting down...")
    if lucid_controller:
        lucid_controller.cleanup()
    logger.info("Cleanup complete. Goodbye!")

# Note: For running this server, you'd typically use a runner script that calls uvicorn,
# or if running this file directly:
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=server_config.get("unified_server_port", 8080))