import os
import requests
from lucid_memory.memory_node import MemoryNode
from typing import Optional, List, Dict, Any
import re
import yaml
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
DEBUG = True
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "proxy_config.json")
PROMPTS_FILE_PATH = os.path.join(os.path.dirname(__file__), "prompts.yaml")


class Digestor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load config if not provided
        if config is None:
             if not os.path.exists(CONFIG_PATH): raise FileNotFoundError(f"LLM config missing: {CONFIG_PATH}")
             with open(CONFIG_PATH, "r", encoding="utf-8") as f: config = json.load(f)
             logging.info(f"Digestor loaded its own config from {CONFIG_PATH}") 
        
        self.llm_url = config.get("backend_url")
        self.model_name = config.get("model_name")
        if not self.llm_url or not self.model_name: raise ValueError("LLM config needs URL/Model.")
        
        self.prompts: Dict[str, str] = {}
        try: # Load Prompts
            with open(PROMPTS_FILE_PATH, "r", encoding="utf-8") as f: loaded_prompts = yaml.safe_load(f)
            # Check required keys - now including code specific ones
            required_keys = ['summary', 'key_concepts', 'tags', 'questions', 'code_dependencies', 'code_outputs']
            if not isinstance(loaded_prompts, dict) or not all(k in loaded_prompts for k in required_keys):
                missing = [k for k in required_keys if k not in (loaded_prompts or {})]
                raise ValueError(f"Prompts YAML missing required keys: {missing}")
            self.prompts = loaded_prompts
            logging.info(f"Digestor: Loaded prompts from {PROMPTS_FILE_PATH}")
        # ... Error handling for prompt loading ...
        except FileNotFoundError:
            logging.error(f"FATAL: Prompts missing: {PROMPTS_FILE_PATH}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"FATAL: YAML parse error: {e}", exc_info=True)
            raise
        except Exception as e:
            logging.error(f"FATAL: Load prompts failed: {e}", exc_info=True)
            raise
        logging.info("Digestor initialized.")

    def _format_prompt(self, key: str, **kwargs) -> Optional[str]:
        # ... Formats template from self.prompts ...
        template = self.prompts.get(key)
        if not template: logging.error(f"Error: Prompt key '{key}' not found."); return None
        try: return template.format(**kwargs)
        except KeyError as e: logging.error(f"Error: Prompt template '{key}' missing placeholder: {e}"); return None

    def _call_llm(self, prompt: str, task_description: str) -> Optional[str]:
        # ... Makes LLM call, returns stripped string content or None ...
        # ... Includes timeout=180 ...
        if not prompt : return None # Don't call if prompt formatting failed
        if DEBUG: logging.info(f"\n--- Calling LLM for: {task_description} ---"); # Log prompt? logging.debug(prompt)
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
                        # Basic cleaning for common LLM non-content additions
                        if content.startswith("YOUR ANALYSIS:") : content = content.split("YOUR ANALYSIS:", 1)[-1].strip()
                        if content.startswith("```"): content = content.split("\n", 1)[-1] # Remove ```json line maybe
                        if content.endswith("```"): content = content.rsplit("\n", 1) # Remove ``` line maybe
                        content = content.strip()
                        if DEBUG: logging.info(f"--- LLM Raw Response ({task_description}) ---\n{content}\n" + "-"*30)
                        return content
            logging.error(f"Error: Unexpected LLM format for {task_description}: {data}")
            return None
        except requests.exceptions.Timeout: logging.error(f"Error: LLM call timed out (180s) for {task_description}."); return None
        except requests.exceptions.RequestException as e: logging.error(f"Error contacting LLM backend ({task_description}) @ {self.llm_url}: {e}"); return None
        except Exception as e: logging.error(f"LLM call error ({task_description}): {e}", exc_info=True); return None

    def _parse_list_output(self, raw_output: Optional[str], task_desc: str) -> List[str]:
        """Parses newline or comma separated list from LLM, cleans items."""
        if not raw_output or raw_output.strip().lower() == "none":
            return []
        # Try splitting by newline first, then comma
        items = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if len(items) <= 1 and ',' in raw_output: # Maybe it used commas?
             items = [item.strip() for item in raw_output.split(',') if item.strip()]

        # Basic cleaning (remove leading bullets/numbers, quotes)
        cleaned_items = []
        for item in items:
             item = re.sub(r"^\s*[-\*\d\.]+\s*", "", item) # Remove list markers like -, *, 1.
             item = re.sub(r'^["\']|["\']$', '', item)    # Remove surrounding quotes
             if item.strip().lower() != "none" and item.strip(): # Filter out empty/None after cleaning
                 cleaned_items.append(item.strip())

        if not cleaned_items:
             logging.warning(f"Warning: Could not parse list output for {task_desc}. Raw: '{raw_output}'")

        return cleaned_items


    def digest(self,
               raw_text: str, # Now receiving the raw chunk text
               node_id: str,
               # Extract chunk_type from the node_id prefix if possible
               chunk_metadata: Optional[Dict[str, Any]] = None, # Or pass metadata in
               generate_questions: bool = False
               ) -> Optional[MemoryNode]:
        """
        Digests a text or code chunk using multiple LLM calls.
        Uses specific prompts for code I/O if chunk_type indicates code.
        """
        if chunk_metadata is None: chunk_metadata = {} # Ensure it's a dict

        # Determine chunk type (example - refine as needed)
        chunk_type = chunk_metadata.get('type', 'unknown')
        is_code_chunk = chunk_type.startswith('python_')

        logging.info(f"\n--- Starting Digestion (Type: {chunk_type}, Questions: {generate_questions}) for Node: {node_id} ---")

        # 1. Get Summary (Universal)
        summary_prompt = self._format_prompt('summary', raw_text=raw_text)
        summary_raw = self._call_llm(summary_prompt, "Summary Generation") if summary_prompt else None
        # --- Clean Summary ---
        summary = f"Summary unavailable: {node_id}";
        if summary_raw:
            lines = [line.strip() for line in summary_raw.splitlines() if line.strip()]; summary = lines[0] if lines else summary_raw
            prefix = next((p for p in ["Summary:","SUMMARY SENTENCE:"] if summary.lower().startswith(p.lower())), None)
            if prefix: summary = summary[len(prefix):].strip()
            if not summary: logging.warning("Summary empty after cleaning."); summary = summary_raw # Fallback
        else: logging.warning(f"Failed summary response node {node_id}.")


        # 2. Get Key Concepts / Logical Steps (Universal for now)
        concepts_prompt = self._format_prompt('key_concepts', raw_text=raw_text) ## <-- USING KCs FOR BOTH NOW
        concepts_raw = self._call_llm(concepts_prompt, "Key Concept Extraction") if concepts_prompt else None
        key_concepts = self._parse_list_output(concepts_raw, f"Key Concepts ({node_id})")
        if not concepts_raw: logging.warning(f"Failed concept response node {node_id}.")


        # 3. Get Tags (Universal)
        tags_prompt = self._format_prompt('tags', raw_text=raw_text)
        tags_raw = self._call_llm(tags_prompt, "Tag Generation") if tags_prompt else None
        tags = self._parse_list_output(tags_raw, f"Tags ({node_id})")
        if not tags_raw: logging.warning(f"Failed tags response node {node_id}.")


        # 4. Code-Specific: Dependencies and Outputs
        dependencies: List[str] = []
        produced_outputs: List[str] = []
        if is_code_chunk:
            # --- Call 4a: Dependencies ---
            dep_prompt = self._format_prompt('code_dependencies', code_chunk=raw_text)
            dep_raw = self._call_llm(dep_prompt, "Code Dependency Extraction") if dep_prompt else None
            dependencies = self._parse_list_output(dep_raw, f"Code Dependencies ({node_id})")
            if not dep_raw: logging.warning(f"Failed dependencies response {node_id}.")

            # --- Call 4b: Outputs (pass dependencies contextually) ---
            dep_list_str = "\n".join([f"- {d}" for d in dependencies]) if dependencies else "(None identified)"
            out_prompt = self._format_prompt('code_outputs', code_chunk=raw_text, dependency_list=dep_list_str)
            out_raw = self._call_llm(out_prompt, "Code Output Extraction") if out_prompt else None
            produced_outputs = self._parse_list_output(out_raw, f"Code Outputs ({node_id})")
            if not out_raw: logging.warning(f"Failed outputs response {node_id}.")

        # 5. Follow-up Questions (Optional, Universal)
        follow_up_questions: List[str] = []
        if generate_questions:
            # ... (Question generation logic remains same as before) ...
            q_prompt = self._format_prompt('questions', raw_text=raw_text)
            q_raw = self._call_llm(q_prompt, "Follow-up Question Generation") if q_prompt else None
            if q_raw: follow_up_questions = self._parse_list_output(q_raw, f"Follow-up Questions ({node_id})")
            if not q_raw: logging.warning(f"Failed question response {node_id}.")
        else:
             logging.info("Skipping follow-up question generation.")

        logging.info(f"------ Finished Digestion for {node_id} ------")

        # Assemble includes structural links which are added by the processor *later*
        return MemoryNode(
            id=node_id,
            raw=raw_text,
            summary=summary,
            key_concepts=key_concepts,
            tags=tags,
            # Linking fields added by Processor
            # sequence_index=None, # Will be populated by Processor
            # parent_identifier=None, # Will be populated by Processor
            # Code specific fields (empty if not code)
            dependencies=dependencies,
            produced_outputs=produced_outputs,
            # internal_vars=[], # Not extracting this anymore
            follow_up_questions=follow_up_questions
            # source=None # Processor could add this too
        )