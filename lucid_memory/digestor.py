import os
import requests
from lucid_memory.memory_node import MemoryNode
from typing import Optional, List, Dict, Any
import re
import yaml
import logging
import json
import time # For potential retries or delays

logger = logging.getLogger(__name__)

DEBUG_DIGESTOR_LLM_CALLS = True # Class-level or instance-level debug flag
LUCID_MEMORY_DIR_DIGESTOR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH_DIGESTOR = os.path.join(LUCID_MEMORY_DIR_DIGESTOR, "proxy_config.json")
DEFAULT_PROMPTS_FILE_PATH_DIGESTOR = os.path.join(LUCID_MEMORY_DIR_DIGESTOR, "prompts.yaml")


class Digestor:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.llm_url: Optional[str] = None
        self.model_name: Optional[str] = None
        self.api_key: Optional[str] = None
        self.prompts: Dict[str, str] = {}
        self.debug_calls = DEBUG_DIGESTOR_LLM_CALLS

        if config is None:
            logger.warning("Digestor initialized without explicit config. Attempting to load default.")
            if not os.path.exists(DEFAULT_CONFIG_PATH_DIGESTOR):
                raise FileNotFoundError(f"Digestor default LLM config missing: {DEFAULT_CONFIG_PATH_DIGESTOR}")
            try:
                with open(DEFAULT_CONFIG_PATH_DIGESTOR, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info(f"Digestor loaded its own config from {DEFAULT_CONFIG_PATH_DIGESTOR}")
            except Exception as e:
                raise ValueError(f"Failed to load default config for Digestor: {e}")
        
        if not isinstance(config, dict):
            raise ValueError("Digestor config must be a dictionary.")

        self.llm_url = config.get("backend_url")
        self.model_name = config.get("model_name")
        self.api_key = config.get("api_key") # Store API key

        if not self.llm_url or not self.model_name:
            raise ValueError("Digestor LLM config must include 'backend_url' and 'model_name'.")
        
        try: 
            with open(DEFAULT_PROMPTS_FILE_PATH_DIGESTOR, "r", encoding="utf-8") as f:
                loaded_prompts = yaml.safe_load(f)
            
            required_keys = ['summary', 'key_concepts', 'tags', 'questions', 
                             'code_dependencies', 'code_outputs', 'code_key_variables'] # ADDED code_key_variables
            if not isinstance(loaded_prompts, dict) or not all(k in loaded_prompts for k in required_keys):
                missing = [k for k in required_keys if k not in (loaded_prompts or {})]
                raise ValueError(f"Prompts YAML '{DEFAULT_PROMPTS_FILE_PATH_DIGESTOR}' missing required keys: {missing}")
            self.prompts = loaded_prompts
            logger.info(f"Digestor: Loaded prompts successfully from {DEFAULT_PROMPTS_FILE_PATH_DIGESTOR}")
        except FileNotFoundError:
            logger.critical(f"FATAL: Prompts file missing for Digestor: {DEFAULT_PROMPTS_FILE_PATH_DIGESTOR}")
            raise
        except yaml.YAMLError as e:
            logger.critical(f"FATAL: Prompts YAML parse error in '{DEFAULT_PROMPTS_FILE_PATH_DIGESTOR}': {e}", exc_info=True)
            raise
        except Exception as e: # Catch-all for other prompt loading issues
            logger.critical(f"FATAL: Failed to load prompts for Digestor: {e}", exc_info=True)
            raise
        logger.info(f"Digestor initialized. LLM URL: {self.llm_url}, Model: {self.model_name}, API Key Present: {bool(self.api_key)}")

    def _format_prompt(self, key: str, **kwargs) -> Optional[str]:
        template = self.prompts.get(key)
        if not template:
            logger.error(f"Digestor Error: Prompt template key '{key}' not found in loaded prompts.")
            return None
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logger.error(f"Digestor Error: Prompt template '{key}' is missing a placeholder: {e}. Args provided: {kwargs.keys()}")
            return None

    def _call_llm(self, prompt: str, task_description: str, temperature: float = 0.1, max_retries: int = 1) -> Optional[str]:
        if not prompt:
            logger.warning(f"Digestor: LLM call for '{task_description}' skipped due to empty prompt.")
            return None
        
        if self.debug_calls:
            logger.info(f"\n--- Digestor Calling LLM for: {task_description} ---\nPrompt (first 200 chars): {prompt[:200]}...\n---")

        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.api_key: # Add API key to header if present
            headers['Authorization'] = f"Bearer {self.api_key}"
            if self.debug_calls: logger.info(f"Digestor LLM call for {task_description} includes Authorization header.")

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "temperature": temperature # Allow configurable temperature
            # "max_tokens": ... consider adding if responses are too long/short
        }
        
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(self.llm_url, headers=headers, json=payload, timeout=180) # 3 min timeout
                response.raise_for_status() # Raises HTTPError for 4xx/5xx
                
                data = response.json()
                choices = data.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str):
                            content = content.strip()
                            # More robust cleaning for common LLM non-content additions
                            prefixes_to_strip = ["YOUR ANALYSIS:", "RESPONSE:", "RESULT:"]
                            for prefix in prefixes_to_strip:
                                if content.upper().startswith(prefix):
                                    content = content[len(prefix):].strip()
                            
                            # Remove markdown code blocks if they surround the whole response
                            if content.startswith("```") and content.endswith("```"):
                                lines = content.splitlines()
                                if len(lines) > 1: # Keep content if it's just one line like ```value```
                                    content = "\n".join(lines[1:-1]).strip() if len(lines) > 2 else lines[0][3:-3].strip()
                            
                            if self.debug_calls:
                                logger.info(f"--- Digestor LLM Raw Response ({task_description}, Attempt {attempt+1}) ---\n{content}\n" + "-"*30)
                            return content
                
                logger.error(f"Digestor Error: Unexpected LLM response format for {task_description} (Attempt {attempt+1}): {str(data)[:200]}...")

            except requests.exceptions.Timeout:
                logger.warning(f"Digestor Error: LLM call timed out (180s) for {task_description} (Attempt {attempt+1}/{max_retries+1}).")
                if attempt == max_retries: logger.error(f"LLM call for {task_description} failed after all retries (Timeout).")
            except requests.exceptions.RequestException as e: # Covers ConnectionError, HTTPError, etc.
                status_code = e.response.status_code if e.response is not None else "N/A"
                logger.warning(f"Digestor Error: LLM call failed for {task_description} (Status: {status_code}, Attempt {attempt+1}/{max_retries+1}): {e}")
                if attempt == max_retries: logger.error(f"LLM call for {task_description} failed after all retries (RequestException).")
            except Exception as e: # Catch-all for other unexpected errors like JSONDecodeError
                logger.error(f"Digestor Error: Unexpected issue during LLM call for {task_description} (Attempt {attempt+1}): {e}", exc_info=True)

            if attempt < max_retries:
                time.sleep(2 ** attempt) # Exponential backoff (2s, 4s, ...)
            else:
                return None # All retries failed
        return None # Should be unreachable if loop logic is correct

    def _parse_list_output(self, raw_output: Optional[str], task_desc: str) -> List[str]:
        if not raw_output or raw_output.strip().lower() in ["none", "n/a", "not applicable"]:
            return []
        
        # Try splitting by newline first, then comma if newlines don't yield multiple items
        items = [line.strip() for line in raw_output.splitlines() if line.strip()]
        if len(items) <= 1 and ',' in raw_output:
             items = [item.strip() for item in raw_output.split(',') if item.strip()]

        cleaned_items = []
        for item in items:
             # Remove leading list markers (hyphens, asterisks, numbers with dots/parentheses)
             item = re.sub(r"^\s*([-*\d]+[\.$]?\s*)+", "", item)
             # Remove surrounding quotes (single or double)
             item = re.sub(r'^["\'](.*)["\']$', r'\1', item) # Non-greedy match inside quotes
             item = item.strip()
             if item and item.lower() not in ["none", "n/a", "not applicable"]:
                 cleaned_items.append(item)

        if not cleaned_items and raw_output.strip(): # If parsing failed but there was input
             logger.warning(f"Digestor Warning: Could not parse any valid items from list output for {task_desc}. Raw: '{raw_output[:100]}...'")
             # As a last resort, if it's a single, non-empty item after basic strip, take it
             single_item_attempt = raw_output.strip()
             if single_item_attempt and single_item_attempt.lower() not in ["none", "n/a", "not applicable"]:
                 return [single_item_attempt]


        return cleaned_items


    def digest(self,
               raw_text: str,
               node_id: str,
               chunk_metadata: Optional[Dict[str, Any]] = None,
               generate_questions: bool = False
               ) -> Optional[MemoryNode]:
        
        if not raw_text.strip():
            logger.warning(f"Digestor: Received empty raw_text for node_id {node_id}. Skipping digestion.")
            return None
            
        if chunk_metadata is None: chunk_metadata = {} 

        chunk_type = chunk_metadata.get('type', 'unknown_text_type')
        is_code_chunk = chunk_type.startswith(('python_', 'py_')) # Broader check for code

        logger.info(f"\n--- Digestor Starting Digestion for Node: {node_id} (Type: {chunk_type}, Length: {len(raw_text)}, GenQuestions: {generate_questions}) ---")

        # 1. Get Summary (Universal)
        summary_prompt = self._format_prompt('summary', raw_text=raw_text)
        summary_raw = self._call_llm(summary_prompt, f"Summary ({node_id})")
        summary = f"Summary unavailable for node: {node_id}" # Fallback
        if summary_raw:
            lines = [line.strip() for line in summary_raw.splitlines() if line.strip()]
            summary = lines[0] if lines else summary_raw # Take first non-empty line as summary
            summary = summary.replace("\"", "") # Remove quotes that LLMs sometimes add
            if not summary.strip(): # If cleaning results in empty, revert to raw or a standard message
                logger.warning(f"Node {node_id}: Summary became empty after cleaning. Raw LLM output: '{summary_raw[:100]}...'")
                summary = f"Content summary for {node_id}." # Generic fallback
        else:
            logger.warning(f"Node {node_id}: Failed to get summary from LLM.")

        # 2. Get Key Concepts / Logical Steps (Prompt differs for code vs text)
        prompt_key_for_concepts = 'code_key_variables' if is_code_chunk else 'key_concepts' # MODIFIED: Use 'code_key_variables' for code
        concepts_prompt_text = self._format_prompt(prompt_key_for_concepts, raw_text=raw_text if not is_code_chunk else "", code_chunk=raw_text if is_code_chunk else "")
        
        concepts_raw = self._call_llm(concepts_prompt_text, f"Key Concepts/Variables ({node_id})", temperature=0.0) # Low temp for extraction
        key_concepts = self._parse_list_output(concepts_raw, f"Key Concepts/Variables ({node_id})")
        if not key_concepts and is_code_chunk:
             logger.warning(f"Node {node_id} (code): No key variables/params extracted.")
        elif not key_concepts:
            logger.warning(f"Node {node_id} (text): No key concepts extracted.")


        # 3. Get Tags (Universal)
        tags_prompt = self._format_prompt('tags', raw_text=raw_text)
        tags_raw = self._call_llm(tags_prompt, f"Tags ({node_id})", temperature=0.0)
        tags = self._parse_list_output(tags_raw, f"Tags ({node_id})")
        if not tags: logger.warning(f"Node {node_id}: No tags extracted.")

        # 4. Code-Specific: Dependencies and Outputs
        dependencies: List[str] = []
        produced_outputs: List[str] = []
        if is_code_chunk:
            dep_prompt = self._format_prompt('code_dependencies', code_chunk=raw_text)
            dep_raw = self._call_llm(dep_prompt, f"Code Dependencies ({node_id})", temperature=0.0)
            dependencies = self._parse_list_output(dep_raw, f"Code Dependencies ({node_id})")
            if not dependencies: logger.info(f"Node {node_id}: No specific code dependencies identified by LLM.")

            dep_list_str = "\n".join([f"- {d}" for d in dependencies]) if dependencies else "(None explicitly identified as dependencies)"
            out_prompt = self._format_prompt('code_outputs', code_chunk=raw_text, dependency_list=dep_list_str)
            out_raw = self._call_llm(out_prompt, f"Code Outputs ({node_id})", temperature=0.0)
            produced_outputs = self._parse_list_output(out_raw, f"Code Outputs ({node_id})")
            if not produced_outputs: logger.info(f"Node {node_id}: No specific code outputs identified by LLM.")

        # 5. Follow-up Questions (Optional, Universal)
        follow_up_questions: List[str] = []
        if generate_questions:
            q_prompt = self._format_prompt('questions', raw_text=raw_text)
            q_raw = self._call_llm(q_prompt, f"Follow-up Questions ({node_id})", temperature=0.3) # Slightly higher temp for creative task
            if q_raw:
                follow_up_questions = self._parse_list_output(q_raw, f"Follow-up Questions ({node_id})")
            if not follow_up_questions: logger.info(f"Node {node_id}: No follow-up questions generated.")
        else:
             logger.debug(f"Node {node_id}: Skipping follow-up question generation as per request.")

        logger.info(f"------ Digestor Finished Digestion for Node: {node_id} ------")

        return MemoryNode(
            id=node_id,
            raw=raw_text,
            summary=summary,
            key_concepts=key_concepts,
            tags=tags,
            dependencies=dependencies,
            produced_outputs=produced_outputs,
            follow_up_questions=follow_up_questions
            # sequence_index, parent_identifier, source, embedding are added by Processor/Controller
        )