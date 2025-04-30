import os
import json # For parsing potential errors
import logging
import requests # Use requests for API call
import time # For timing
from urllib.parse import urljoin, urlparse # To construct embedding URL
from typing import List, Dict, Any, Optional

# Use forward reference 'MemoryNode' in type hints to avoid circular import issues
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .memory_node import MemoryNode

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(processName)s - %(threadName)s - Embedder - %(message)s')

# Standard OpenAI API path relative to base URL
DEFAULT_EMBEDDING_API_PATH = "v1/embeddings" # Standard path

class Embedder:
    """
    Handles the creation of vector embeddings for MemoryNodes using an
    OpenAI-compatible API endpoint (e.g., Ollama, LM Studio, vLLM).
    No local sentence-transformer dependency.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Embedder with backend configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary, minimally expecting
                                     'backend_url' and 'model_name'.
        """
        self.backend_url_raw: Optional[str] = config.get("backend_url")
        # Model name to REQUEST from the API
        self.api_model_name: Optional[str] = config.get("model_name")
        # Derived endpoint URL for /v1/embeddings
        self.embedding_endpoint_url: Optional[str] = None
        # Simple readiness flag - True if base URL parses AND model name present
        self.is_configured: bool = False

        if not self.backend_url_raw or not self.api_model_name:
             logging.error("Config MISSING required 'backend_url' or 'model_name'. Embedder DISABLED.")
             return # Cannot function without these

        try:
            # Reliably derive BASE server URL (e.g., http://host:port)
            parsed_url = urlparse(self.backend_url_raw)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid backend_url format: '{self.backend_url_raw}'")
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

            # Construct the full embedding endpoint URL relative to the base
            self.embedding_endpoint_url = urljoin(base_url, DEFAULT_EMBEDDING_API_PATH)

            if not self.embedding_endpoint_url:
                   raise ValueError("URL joining failed unexpectedly.")

            logging.info(f"Embedder configured: Endpoint='{self.embedding_endpoint_url}', Model='{self.api_model_name}'")
            self.is_configured = True # Config seems okay

        except ValueError as e:
            logging.error(f"Invalid backend_url ('{self.backend_url_raw}') prevents deriving embedding endpoint: {e}. Embedder DISABLED.")
            self.embedding_endpoint_url = None
        except Exception as e: # Catch any other unexpected errors during init
            logging.exception(f"Unexpected error initializing Embedder from config: {e}. Embedder DISABLED.")
            self.embedding_endpoint_url = None


    def is_available(self) -> bool:
        """
        Checks if Embedder configuration was successful.
        Does not guarantee API reachability or model validity on backend.
        """
        return self.is_configured and self.embedding_endpoint_url is not None

    def _prepare_text_for_embedding(self, node: 'MemoryNode') -> str:
        """
        Constructs the text string from node fields for embedding.
        (Logic remains same - combining summary, concepts, IO)
        """
        # Delay import to method scope if needed, but should typically be fine unless complex deps
        from .memory_node import MemoryNode

        if not isinstance(node, MemoryNode):
            logging.warning("Embedder received unexpected object type during text prep.")
            return ""

        parts = []
        if node.summary and node.summary.strip():
            parts.append(node.summary.strip())
        if node.key_concepts:
            parts.append("CONCEPTS: " + ", ".join(node.key_concepts).strip())
        if node.dependencies:
            parts.append("REQUIRES: " + ", ".join(node.dependencies).strip())
        if node.produced_outputs:
            parts.append("PRODUCES: " + ", ".join(node.produced_outputs).strip())

        combined_text = " | ".join(filter(None, parts)) # Filter out empty strings

        if not combined_text:
            # Use getattr for safe access to id in case node is malformed
            node_id = getattr(node, 'id', 'UNKNOWN_NODE')
            logging.warning(f"Node {node_id}: No text content could be prepared for embedding. Returning empty string.")
            return ""

        # Debug log the prepared text (truncated)
        logging.debug(f"PrepText Node {getattr(node, 'id', '?')}: '{combined_text[:150]}...' (len={len(combined_text)})")
        return combined_text


    def generate_embedding(self, node: 'MemoryNode') -> Optional[List[float]]:
        """
        Generates vector embedding using the configured OpenAI-compatible API endpoint.

        Args:
            node (MemoryNode): The node containing data to embed.

        Returns:
            Optional[List[float]]: Embedding vector list, or None on failure.
        """
        node_id = getattr(node, 'id', 'UNKNOWN_NODE') # Safe ID access for logs

        if not self.is_available():
            logging.warning(f"Embedder not configured/available, skipping embedding for Node {node_id}.")
            return None
        if not node: # Basic guard
            logging.warning("Embedder received 'None' instead of a Node object.")
            return None

        # --- Prepare Text ---
        text_to_embed = ""
        try:
            text_to_embed = self._prepare_text_for_embedding(node)
            if not text_to_embed:
                 logging.error(f"Failed to prepare text for Node {node_id}, cannot generate embedding.")
                 return None
        except Exception as prep_err:
            # Catch errors specifically during text prep (e.g., unexpected node data type)
            logging.exception(f"Error PREPARING text for embedding Node {node_id}: {prep_err}")
            return None

        # --- Prepare API Request ---
        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        payload = {
            "input": text_to_embed,
            "model": self.api_model_name # Use the NAME configured
            # NOTE: Some backends might support other params like "encoding_format": "float"
            # Keep payload simple for broad compatibility first.
        }

        emb_vector: Optional[List[float]] = None # Explicitly define type and init
        start_time = time.monotonic()
        timeout_seconds = 60 # Configurable? Embedding can be slow for large models/texts

        # --- Make API Call ---
        try:
            logging.debug(f"POSTing to {self.embedding_endpoint_url} for Node {node_id} (Model: {self.api_model_name}, TextLen: {len(text_to_embed)})")

            # Ensure endpoint URL is valid before request
            if not self.embedding_endpoint_url:
                 logging.critical(f"PANIC: Embedding endpoint URL became invalid before API call for {node_id}!")
                 return None

            response = requests.post(
                self.embedding_endpoint_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds
            )

            response.raise_for_status() # Checks for 4xx/5xx HTTP errors

            # --- Parse GOOD Response (2xx) ---
            response_data = response.json() # Parse JSON *after* checking status code

            # Validate response structure (OpenAI API like)
            if not isinstance(response_data, dict):
                raise TypeError(f"API response is not a JSON dictionary (got {type(response_data)}).")

            if 'data' not in response_data or not isinstance(response_data['data'], list) or not response_data['data']:
                raise ValueError("API response missing 'data' list or 'data' is empty.")

            first_item = response_data['data'][0] # Assume embedding for single input is first item
            if not isinstance(first_item, dict) or 'embedding' not in first_item:
                raise ValueError("First item in 'data' list is not a dict or misses 'embedding' key.")

            embedding_val = first_item['embedding']
            if not isinstance(embedding_val, list):
                 raise TypeError(f"Value of 'embedding' key is not a list (got {type(embedding_val)}).")

            # Final check: Ensure list contains numbers (float or int)
            if not all(isinstance(x, (float, int)) for x in embedding_val):
                 raise ValueError("Embedding list contains non-numeric values.")

            # If all checks pass, assign the vector
            emb_vector = embedding_val # Successfully extracted

        # --- Handle Specific Request Errors ---
        except requests.exceptions.Timeout:
            logging.error(f"TIMEOUT ({timeout_seconds}s) connecting to Embedding API '{self.embedding_endpoint_url}' for Node {node_id}.")
        except requests.exceptions.ConnectionError:
            logging.error(f"CONNECTION ERROR reaching Embedding API '{self.embedding_endpoint_url}' for Node {node_id}. Backend down?")
        except requests.exceptions.RequestException as http_err:
            # Includes HTTP errors raised by raise_for_status()
            status_code = http_err.response.status_code if http_err.response is not None else "N/A"
            err_text = http_err.response.text[:200] if http_err.response is not None else "(No response body)"
            logging.error(f"HTTP Error ({status_code}) from Embedding API '{self.embedding_endpoint_url}' for Node {node_id}: {err_text}", exc_info=False)
        # --- Handle JSON Parsing / Structure Errors ---
        except (json.JSONDecodeError, TypeError, ValueError) as parse_err:
             # Logged based on where error occurred (JSON parse or Value/Type validation)
            logging.error(f"Error PARSING/Validating response from Embedding API for Node {node_id}: {parse_err}", exc_info=False)
            # Debug log raw response if useful
            if 'response' in locals() and hasattr(response, 'text'): logging.debug(f"RAW RESPONSE TEXT: {response.text[:500]}...")
        # --- Catch ANY other unexpected error ---
        except Exception as general_err:
            logging.exception(f"UNEXPECTED error during embedding generation for Node {node_id}: {general_err}") # Log full traceback


        # --- Final Logging & Return ---
        end_time = time.monotonic()
        duration = end_time - start_time

        if emb_vector is not None:
            dim = len(emb_vector)
            logging.info(f"Successfully generated embedding for Node {node_id} (Dim: {dim}, Time: {duration:.3f}s)")
            return emb_vector # Return the valid list of floats
        else:
            logging.warning(f"FAILED to generate embedding for Node {node_id} (Time: {duration:.3f}s). See previous errors.")
            return None # Return None on any failure path