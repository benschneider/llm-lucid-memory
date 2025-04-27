from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.retriever import ReflectiveRetriever
from lucid_memory.digestor import Digestor
import json
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "proxy_config.json")

# Load config
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
        BACKEND_URL = config.get("backend_url", "http://localhost:11434/v1/chat/completions")
        MODEL_NAME = config.get("model_name", "mistral")
else:
    BACKEND_URL = "http://localhost:11434/v1/chat/completions"
    MODEL_NAME = "mistral"

app = FastAPI()


# Load memory components
digestor = Digestor()
memory_graph = MemoryGraph()
retriever = ReflectiveRetriever(memory_graph)

# Define Request format
class ChatRequest(BaseModel):
    messages: list
    temperature: float = 0.2

@app.post("/chat")
async def smart_chat(request: ChatRequest):
    user_message = request.messages[-1]["content"]

    # Retrieve relevant memories
    candidates = retriever.retrieve_by_keyword(user_message)
    best_nodes = retriever.reflect_on_candidates(candidates, user_message)

    # Build enhanced prompt
    memory_prompt = ""
    for i, memory in enumerate(best_nodes, 1):
        memory_prompt += f"Memory {i}:\nSummary: {memory.summary}\nReasoning Paths:\n"
        for rp in memory.reasoning_paths:
            memory_prompt += f"- {rp}\n"
        memory_prompt += "\n"

    full_prompt = memory_prompt + f"\nQuestion:\n{user_message}\n\nPlease reason using the memories provided. Draft the steps logically."

    # Prepare request to external LLM backend
    system_prompt = """You are a reasoning assistant.

You MUST ONLY use the provided memory reasoning points to answer.

Format your answer as a minimal logical chain of steps, not full verbose sentences.

Example:
Memories:
- Open socket
- Bind port
- Accept connection
- Perform TLS handshake
- Serve secured content

Question:
How does the server start and accept secure connections?

Answer:
open socket -> bind port -> accept connection -> TLS handshake -> serve secured content

Keep it compact, meaningful, and ordered logically based on the memories.
"""

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ],
        "temperature": request.temperature,
    }

    response = requests.post(BACKEND_URL, json=payload)
    return response.json()