import os
import json
import logging
import requests
import time
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Any, Optional
from .memory_node import MemoryNode


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - Embedder - %(message)s')

DEFAULT_EMBEDDING_API_PATH = "v1/embeddings"
DEFAULT_EMBEDDING_MODEL_FALLBACK = "nomic-embed-text"


class Embedder:
    def __init__(self, config: Dict[str, Any]):
        self.backend_url_raw: Optional[str] = config.get("backend_url")
        self.api_model_name: Optional[str] = config.get("embedding_api_model_name", DEFAULT_EMBEDDING_MODEL_FALLBACK)
        if not self.api_model_name:
            logging.warning("No embedding model name configured and fallback is empty. Embedder might not work.")
            self.api_model_name = DEFAULT_EMBEDDING_MODEL_FALLBACK

        self.embedding_endpoint_url: Optional[str] = None
        self.is_configured: bool = False
        self.api_key: Optional[str] = config.get("api_key")

        if not self.backend_url_raw:
             logging.error("Embedder Config MISSING 'backend_url'. Embedder DISABLED.")
             return

        try:
            parsed_url = urlparse(self.backend_url_raw)
            if not parsed_url.scheme or not parsed_url.netloc:
                raise ValueError(f"Invalid backend_url format for Embedder: '{self.backend_url_raw}'")
            base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
            self.embedding_endpoint_url = urljoin(base_url, DEFAULT_EMBEDDING_API_PATH)

            if not self.embedding_endpoint_url:
                   raise ValueError("URL joining failed for embedding endpoint.")

            logging.info(f"Embedder configured: Endpoint='{self.embedding_endpoint_url}', Model='{self.api_model_name}', API Key Present: {bool(self.api_key)}")
            self.is_configured = True

        except ValueError as e:
            logging.error(f"Invalid backend_url ('{self.backend_url_raw}') for Embedder: {e}. Embedder DISABLED.")
            self.embedding_endpoint_url = None
        except Exception as e:
            logging.exception(f"Unexpected error initializing Embedder: {e}. Embedder DISABLED.")
            self.embedding_endpoint_url = None


    def is_available(self) -> bool:
        return self.is_configured and self.embedding_endpoint_url is not None and bool(self.api_model_name)

    def _prepare_text_for_embedding(self, node: MemoryNode) -> str: # Type hint 'MemoryNode' works due to top-level import
        # REMOVED: No local re-import needed for MemoryNode
        # if TYPE_CHECKING: from .memory_node import MemoryNode

        if not isinstance(node, MemoryNode): # This will now use the top-level imported MemoryNode
            logging.warning(f"Embedder received unexpected object type ({type(node)}) instead of MemoryNode during text prep.")
            return ""

        parts = []
        if node.summary and node.summary.strip(): parts.append(f"Summary: {node.summary.strip()}")
        if node.key_concepts: parts.append("Key Concepts/Logic: " + ", ".join(node.key_concepts).strip())
        if node.dependencies: parts.append("Dependencies: " + ", ".join(node.dependencies).strip())
        if node.produced_outputs: parts.append("Outputs: " + ", ".join(node.produced_outputs).strip())
        if node.tags: parts.append("Tags: " + ", ".join(node.tags).strip())
        combined_text = " | ".join(filter(None, parts))

        if not combined_text:
            node_id = getattr(node, 'id', 'UNKNOWN_NODE_ID')
            logging.warning(f"Node {node_id}: No text content could be prepared for embedding. Raw content length: {len(node.raw)}. Using raw content as fallback if not too long.")
            if node.raw and len(node.raw) < 1024 :
                return node.raw.strip()
            else:
                logging.error(f"Node {node_id}: Still no suitable text for embedding after fallback. Raw content too long or empty.")
                return ""
        logging.debug(f"Prepared text for embedding Node {getattr(node, 'id', '?')}: '{combined_text[:150]}...' (Total length: {len(combined_text)})")
        return combined_text


    def generate_embedding(self, node: MemoryNode) -> Optional[List[float]]: # Type hint 'MemoryNode' works
        node_id = getattr(node, 'id', 'UNKNOWN_NODE_ID_IN_GENERATE')

        if not self.is_available():
            logging.warning(f"Embedder not configured/available, skipping embedding for Node {node_id}.")
            return None
        
        # REMOVED: No local re-import needed for MemoryNode
        # if TYPE_CHECKING: from .memory_node import MemoryNode
        if not isinstance(node, MemoryNode): # This will now use the top-level imported MemoryNode
            logging.warning(f"Embedder.generate_embedding received non-MemoryNode object for ID {node_id}.")
            return None

        text_to_embed = ""
        try:
            text_to_embed = self._prepare_text_for_embedding(node)
            if not text_to_embed:
                 logging.error(f"Failed to prepare any text for Node {node_id}, cannot generate embedding.")
                 return None
        except Exception as prep_err:
            logging.exception(f"Error preparing text for embedding Node {node_id}: {prep_err}")
            return None

        if not self.embedding_endpoint_url or not self.api_model_name:
            logging.error(f"Embedder misconfiguration: endpoint or model name is None for Node {node_id}.")
            return None

        headers = {'Content-Type': 'application/json', 'Accept': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f"Bearer {self.api_key}"
            logging.debug(f"Embedding API call for Node {node_id} will include Authorization header.")

        payload = { "input": text_to_embed, "model": self.api_model_name }
        emb_vector: Optional[List[float]] = None
        start_time = time.monotonic()
        timeout_seconds = 60

        try:
            logging.debug(f"POSTing to embedding endpoint {self.embedding_endpoint_url} for Node {node_id} (Model: {self.api_model_name}, TextLen: {len(text_to_embed)})")
            response = requests.post(
                self.embedding_endpoint_url,
                headers=headers,
                json=payload,
                timeout=timeout_seconds
            )
            response.raise_for_status()
            response_data = response.json()

            if not isinstance(response_data, dict):
                raise TypeError(f"API response for embedding is not a JSON dictionary (got {type(response_data)}).")
            if 'data' not in response_data or not isinstance(response_data['data'], list) or not response_data['data']:
                raise ValueError("API response for embedding missing 'data' list or 'data' is empty.")
            first_item = response_data['data'][0]
            if not isinstance(first_item, dict) or 'embedding' not in first_item:
                raise ValueError("First item in embedding 'data' list is not a dict or misses 'embedding' key.")
            embedding_val = first_item['embedding']
            if not isinstance(embedding_val, list):
                 raise TypeError(f"Value of 'embedding' key is not a list (got {type(embedding_val)}).")
            if not all(isinstance(x, (float, int)) for x in embedding_val):
                 if any(x is None for x in embedding_val):
                     raise ValueError("Embedding list contains None values.")
                 raise ValueError("Embedding list contains non-numeric values.")
            emb_vector = [float(x) for x in embedding_val]
        except requests.exceptions.Timeout:
            logging.error(f"TIMEOUT ({timeout_seconds}s) connecting to Embedding API '{self.embedding_endpoint_url}' for Node {node_id}.")
        except requests.exceptions.ConnectionError:
            logging.error(f"CONNECTION ERROR reaching Embedding API '{self.embedding_endpoint_url}' for Node {node_id}. Is the backend service (e.g., Ollama) running and accessible?")
        except requests.exceptions.RequestException as http_err:
            status_code = http_err.response.status_code if http_err.response is not None else "N/A"
            err_text_short = (http_err.response.text[:200] + '...' if hasattr(http_err.response, 'text') and len(http_err.response.text) > 200 else (http_err.response.text if hasattr(http_err.response, 'text') else "(No response body)")) if http_err.response is not None else "(No response object)"
            logging.error(f"HTTP Error ({status_code}) from Embedding API '{self.embedding_endpoint_url}' for Node {node_id}: {err_text_short}", exc_info=False)
            if status_code == 401: logging.error("Hint: Check if API key is required/correct for embedding endpoint.")
        except (json.JSONDecodeError, TypeError, ValueError) as parse_err:
            logging.error(f"Error PARSING/Validating response from Embedding API for Node {node_id}: {parse_err}", exc_info=False)
            if 'response' in locals() and hasattr(response, 'text'): logging.debug(f"RAW RESPONSE TEXT (Embedder error): {response.text[:500]}...")
        except Exception as general_err:
            logging.exception(f"UNEXPECTED error during embedding generation for Node {node_id}: {general_err}")

        end_time = time.monotonic()
        duration = end_time - start_time

        if emb_vector is not None:
            dim = len(emb_vector)
            logging.info(f"Successfully generated embedding for Node {node_id} (Dimensions: {dim}, Time: {duration:.3f}s)")
            return emb_vector
        else:
            logging.warning(f"FAILED to generate embedding for Node {node_id} (Time: {duration:.3f}s). Review previous error logs.")
            return None