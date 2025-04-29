from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import requests
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.retriever import ReflectiveRetriever
import json
import os
import yaml # Added import

# --- Configuration and File Paths ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "proxy_config.json")
MEMORY_GRAPH_PATH = "memory_graph.json"
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompts.yaml") # Added prompts path

# --- Load LLM Config ---
# (Config loading logic remains the same)
# ...
if os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
            BACKEND_URL = config.get("backend_url", "http://localhost:11434/v1/chat/completions")
            MODEL_NAME = config.get("model_name", "mistral")
            print(f"Proxy Config: Loaded backend URL: {BACKEND_URL}, Model: {MODEL_NAME}")
    except Exception as e:
         print(f"Error loading proxy config: {e}. Using defaults.")
         BACKEND_URL = "http://localhost:11434/v1/chat/completions"; MODEL_NAME = "mistral"
else:
    print(f"Proxy config file not found at {CONFIG_PATH}. Using defaults.")
    BACKEND_URL = "http://localhost:11434/v1/chat/completions"; MODEL_NAME = "mistral"

# --- Load Prompts ---
prompts = {}
try:
    with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    if not isinstance(prompts, dict) or 'chat_system_prompt' not in prompts:
         print(f"Warning: Prompts file {PROMPTS_FILE_PATH} is missing 'chat_system_prompt' key. Using default.")
    else:
        print(f"Proxy Server: Successfully loaded prompts from {PROMPTS_FILE_PATH}")
except FileNotFoundError:
    print(f"Warning: Prompts file not found at {PROMPTS_FILE_PATH}. Using default system prompt.")
except yaml.YAMLError as e:
    print(f"Warning: Failed to parse prompts YAML file {PROMPTS_FILE_PATH}: {e}. Using default system prompt.")
except Exception as e:
    print(f"Warning: Failed to load prompts from {PROMPTS_FILE_PATH}: {e}. Using default system prompt.")

# Define a default system prompt as fallback
DEFAULT_CHAT_SYSTEM_PROMPT = "You are a helpful assistant. Answer based on the context provided."

app = FastAPI(title="Lucid Memory Proxy Server")

# --- Memory Component Initialization ---
# (Memory graph loading remains the same)
# ...
memory_graph = MemoryGraph()
if os.path.exists(MEMORY_GRAPH_PATH):
    try:
        memory_graph.load_from_json(MEMORY_GRAPH_PATH)
        print(f"Proxy Server: Loaded {len(memory_graph.nodes)} nodes from {MEMORY_GRAPH_PATH}")
    except Exception as e: print(f"Proxy Server: Error loading memory graph: {e}. Starting empty.")
else: print(f"Proxy Server: Memory graph file not found. Starting empty.")
retriever = ReflectiveRetriever(memory_graph)

# (ChatRequest model remains the same)
class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.2

# --- API Endpoints ---

@app.post("/chat")
async def smart_chat(request: ChatRequest):
    # (Keep initial message extraction logic)
    # ... (extract user_message_content) ...
    if not request.messages: raise HTTPException(status_code=400, detail="Messages list empty.")
    user_message_content = next((msg.get("content", "") for msg in reversed(request.messages) if msg.get("role") == "user"), "")
    if not user_message_content: raise HTTPException(status_code=400, detail="No user message found.")

    print(f"\n--- Received Chat Request ---")
    print(f"User Message: {user_message_content}")

    # 1. Retrieve relevant memories
    # (Retrieval logic remains the same - remember potential TODO for retriever update)
    # ...
    candidates = retriever.retrieve_by_keyword(user_message_content)
    best_nodes = retriever.reflect_on_candidates(candidates, user_message_content)
    print(f"Retrieved {len(candidates)} candidates, selected {len(best_nodes)} best nodes.")


    # 2. Build enhanced prompt with memories
    # (Memory prompt section building remains the same)
    # ...
    memory_prompt_section = ""
    # ... (build memory_prompt_section using best_nodes.summary, .key_concepts, .tags) ...
    if best_nodes:
        memory_prompt_section += "You have the following relevant memories:\n\n"
        for i, memory in enumerate(best_nodes, 1):
            memory_prompt_section += f"--- Memory {i} (ID: {memory.id}) ---\n"
            memory_prompt_section += f"Summary: {memory.summary}\n"
            memory_prompt_section += f"Key Concepts:\n"
            if memory.key_concepts:
                for concept in memory.key_concepts: memory_prompt_section += f"- {concept}\n"
            else: memory_prompt_section += "- (None extracted)\n"
            if memory.tags: memory_prompt_section += f"Tags: {', '.join(memory.tags)}\n"
            memory_prompt_section += "\n"
        memory_prompt_section += "---\n\n"
    else: memory_prompt_section = "No specific relevant memories found for this question.\n\n"

    final_user_prompt = memory_prompt_section + f"Based ONLY on the memories provided (if any), please answer the following question:\n\nQuestion: {user_message_content}"

    # 3. Prepare request to the actual LLM backend
    # --- Use loaded system prompt ---
    system_prompt_content = prompts.get('chat_system_prompt', DEFAULT_CHAT_SYSTEM_PROMPT)
    # --- End Use loaded system prompt ---

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt_content}, # Use the variable
            {"role": "user", "content": final_user_prompt}
        ],
        "temperature": request.temperature,
        "stream": False
    }

    print(f"--- Sending to Backend LLM ({MODEL_NAME} at {BACKEND_URL}) ---")
    print(f"System Prompt: {system_prompt_content[:200]}...") # Log the used prompt
    print(f"User Prompt for LLM: {final_user_prompt[:500]}...")

    # 4. Forward to backend LLM and return response
    try: # (Error handling remains the same)
        response = requests.post(BACKEND_URL, json=payload, timeout=60) # Maybe keep chat timeout lower?
        response.raise_for_status()
        llm_response_data = response.json()
        print(f"--- Received Response from Backend LLM ---")
        return llm_response_data
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="Request to backend LLM timed out.")
    except requests.exceptions.RequestException as e:
        err_detail = f"Failed to communicate with backend LLM: {e}"
        if e.response is not None: err_detail += f" | Response: {e.response.text[:100]}"
        raise HTTPException(status_code=502, detail=err_detail)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# (Keep root endpoint and main block)
@app.get("/")
async def root():
    return { "message": "Lucid Memory Proxy Server is running.", "memory_nodes_loaded": len(memory_graph.nodes), "backend_llm_url": BACKEND_URL, "backend_llm_model": MODEL_NAME }

if __name__ == "__main__":
    import uvicorn
    run_port = 8000 # Default
    if os.path.exists(CONFIG_PATH):
        try: run_port = json.load(open(CONFIG_PATH))['local_proxy_port']
        except Exception: pass
    print(f"Starting Uvicorn server directly on port {run_port}...")
    # Correct way to reference app in uvicorn command for direct run
    module_name = os.path.basename(__file__).replace('.py', '')
    uvicorn.run(f"{module_name}:app", host="0.0.0.0", port=run_port, reload=True)