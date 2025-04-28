# lucid_memory/digestor.py

import os
import json
import requests
from lucid_memory.memory_node import MemoryNode
from typing import Optional, List, Dict, Any # Ensure Dict/Any are imported
import re
import yaml
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEBUG = True
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "proxy_config.json")
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompts.yaml")

class Digestor:
    def __init__(self):
        # (Init remains the same - loads config and self.prompts)
        if not os.path.exists(CONFIG_PATH): raise FileNotFoundError(f"LLM config not found: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: config = json.load(f)
        self.llm_url = config.get("backend_url"); self.model_name = config.get("model_name")
        if not self.llm_url or not self.model_name: raise ValueError("LLM config needs 'backend_url' and 'model_name'.")
        self.prompts: Dict[str, str] = {}
        try:
            with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f: loaded_prompts = yaml.safe_load(f)
            if not isinstance(loaded_prompts, dict) or not all(k in loaded_prompts for k in ['summary', 'key_concepts', 'tags', 'questions']): raise ValueError("Prompts YAML missing required keys.")
            self.prompts = loaded_prompts
            logging.info(f"Digestor: Loaded prompts from {PROMPTS_FILE_PATH}")
        except FileNotFoundError: logging.error(f"FATAL: Prompts file not found: {PROMPTS_FILE_PATH}"); raise
        except yaml.YAMLError as e: logging.error(f"FATAL: YAML parse error: {e}", exc_info=True); raise
        except Exception as e: logging.error(f"FATAL: Failed loading prompts: {e}", exc_info=True); raise
        logging.info("Digestor initialized.")

    # --- NEW: Helper method for formatting prompts ---
    def _format_prompt(self, key: str, **kwargs) -> Optional[str]:
        """Formats a prompt template using provided keyword arguments."""
        template = self.prompts.get(key) # Access instance variable self.prompts
        if not template:
            logging.error(f"Error: Prompt key '{key}' not found in loaded prompts.")
            return None
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logging.error(f"Error: Prompt template '{key}' missing placeholder: {e}")
            return None
    # --- End Helper Method ---

    def _call_llm(self, prompt: str, task_description: str) -> Optional[str]:
        # (This method remains the same)
        if DEBUG: logging.info(f"\n--- Calling LLM for: {task_description} ---")
        payload = { "model": self.model_name, "messages": [{"role": "user", "content": prompt}], "stream": False, "temperature": 0.1 }
        try:
            response = requests.post(self.llm_url, json=payload, timeout=180)
            response.raise_for_status()
            data = response.json()
            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str):
                        content = content.strip()
                        if DEBUG: logging.info(f"--- LLM Raw Response ({task_description}) ---\n{content}\n" + "-"*30)
                        return content
            logging.error(f"Error: Unexpected LLM response format for {task_description}: {data}")
            return None
        except requests.exceptions.Timeout: logging.error(f"Error: LLM call timed out (180s) for {task_description}."); return None
        except requests.exceptions.RequestException as e: logging.error(f"Error contacting LLM backend ({task_description}) @ {self.llm_url}: {e}"); return None
        except Exception as e: logging.error(f"Unexpected error during LLM call ({task_description}): {e}", exc_info=True); return None


    def digest(self, raw_text: str, node_id: str, generate_questions: bool = False) -> Optional[MemoryNode]:
        """Digests raw text using multiple LLM calls defined in prompts.yaml."""
        logging.info(f"\n--- Starting Digestion (Questions: {generate_questions}) for Node: {node_id} ---")

        # --- Updated Calls to use self._format_prompt ---
        summary_prompt = self._format_prompt('summary', raw_text=raw_text)
        concepts_prompt = self._format_prompt('key_concepts', raw_text=raw_text)
        tags_prompt = self._format_prompt('tags', raw_text=raw_text)
        questions_prompt = self._format_prompt('questions', raw_text=raw_text) if generate_questions else None
        # --- End Updates ---

        # 1. Get Summary
        summary_raw = self._call_llm(summary_prompt, "Summary Generation") if summary_prompt else None
        summary = f"Summary unavailable for {node_id}"; # Default
        if summary_raw:
            lines = [line.strip() for line in summary_raw.splitlines() if line.strip()]
            if lines: summary = lines[0]; # Take first non-empty line
            # Clean common prefixes
            prefixes_to_clean = ["Summary:", "SUMMARY SENTENCE:", "summary:"]
            for prefix in prefixes_to_clean:
                 if summary.lower().startswith(prefix.lower()): summary = summary[len(prefix):].strip(); break
            if not summary: logging.warning("Warning: Summary response empty after cleaning."); summary = summary_raw # Fallback if cleaning removed everything
        else: logging.warning("Warning: Failed to get summary response.")

        # 2. Get Key Concepts
        concepts_raw = self._call_llm(concepts_prompt, "Key Concept Extraction") if concepts_prompt else None
        key_concepts: List[str] = []
        if concepts_raw:
            key_concepts = [re.sub(r"^\s*[-\*\d\.]+\s*", "", line).strip() for line in concepts_raw.splitlines() if line.strip()]
            if not key_concepts: logging.warning(f"Warning: Could not parse key concepts. Raw: '{concepts_raw}'")
        else: logging.warning("Warning: Failed to get key concepts response (or timed out).")

        # 3. Get Tags
        tags_raw = self._call_llm(tags_prompt, "Tag Generation") if tags_prompt else None
        tags: List[str] = []
        if tags_raw:
            tags = [tag.strip().lower() for tag in tags_raw.split(',') if tag.strip() and not tag.isspace()]
            tags = [re.sub(r'^[^a-zA-Z0-9\-_]+|[^a-zA-Z0-9\-_]+$', '', tag) for tag in tags if tag] # Allow hyphen/underscore
            tags = list(filter(None, tags))
            if not tags: logging.warning(f"Warning: Could not parse tags. Raw: '{tags_raw}'")
        else: logging.warning("Warning: Failed to get tags response (or timed out).")

        # 4. Get Follow-up Questions (Conditional)
        follow_up_questions: List[str] = []
        if generate_questions and questions_prompt: # Check flag and if prompt exists
            questions_raw = self._call_llm(questions_prompt, "Follow-up Question Generation")
            if questions_raw and questions_raw.strip().lower() != "none":
                follow_up_questions = [ q.strip() for q in questions_raw.splitlines() if q.strip() and q.strip().lower() != "none" ]
                if not follow_up_questions: logging.warning(f"Warning: Could not parse follow-up questions. Raw: '{questions_raw}'")
            elif questions_raw is None: logging.warning("Warning: Failed to get follow-up questions response (or timed out).")
            else: logging.info("Info: LLM indicated no follow-up questions needed.")
        elif generate_questions: # Flag was true but prompt formatting failed
             logging.warning("Warning: Skipped question generation due to prompt formatting error.")
        else: # Flag was false
             logging.info("Skipping follow-up question generation as requested.")

        logging.info(f"--- Finished Digestion for {node_id} ---")

        # Assemble the MemoryNode
        # Assume MemoryNode expects 'key_concepts'. If changed to 'logical_steps', update here.
        return MemoryNode(
            id=node_id,
            raw=raw_text, # Store the original chunk text
            summary=summary,
            key_concepts=key_concepts,
            tags=tags,
            follow_up_questions=follow_up_questions
        )