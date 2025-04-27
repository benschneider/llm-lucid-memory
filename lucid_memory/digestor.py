import os
import json
import requests
from lucid_memory.memory_node import MemoryNode

DEBUG = True

class Digestor:
    def __init__(self):
        config_path = os.path.join(os.path.dirname(__file__), "proxy_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        self.llm_url = config.get("backend_url")
        self.model_name = config.get("model_name")

        if not self.llm_url or not self.model_name:
            raise ValueError("Configuration must contain 'backend_url' and 'model_name'.")

    def digest(self, raw_text: str, node_id: str) -> MemoryNode:
        return self._digest_with_llm(raw_text, node_id)

    def _digest_with_llm(self, raw_text: str, node_id: str) -> MemoryNode:
        prompt = f"""
You are a memory digestor.

Your goal is to deeply understand technical project documents, designs, or code, and produce a compact structured knowledge summary.

Given the following raw text, perform:

- Summarize the main purpose and core ideas in 1-2 sentences.
- Extract 5-10 key concepts, mechanisms, or steps that are central to the system's function or design.
- Identify any open questions, missing details, or follow-up topics that should be explored for complete understanding.

Return ONLY valid JSON with fields:
  - summary (string)
  - key_concepts (list of short sentences)
  - follow_up_items (list of questions or topics to explore)

IMPORTANT: Return ONLY valid JSON. No markdown formatting, no explanations, no prose. 
Your entire response must start with '{{' and end with '}}' — only pure JSON allowed.

TEXT:
\"\"\"
{raw_text}
\"\"\"
"""

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }

        response = requests.post(self.llm_url, json=payload)
        if response.status_code != 200:
            raise RuntimeError(f"LLM digest failed: {response.text}")

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
            if DEBUG:
                print("LLM RAW RESPONSE:", content)
            parsed = json.loads(content)
            return MemoryNode(
                id=node_id,
                raw=raw_text,
                summary=parsed.get("summary", "Summary missing"),
                reasoning_paths=parsed.get("reasoning_paths", []),
                tags=parsed.get("tags", [])
            )
        except Exception as e:
            raise RuntimeError(f"Failed to parse LLM response: {e}")