import logging
import requests
from urllib.parse import urlparse, urljoin
from typing import Optional, List, Dict, Any
from .config_manager import ConfigManager, API_PRESETS

from ..digestor import Digestor
from ..embedder import Embedder

class ComponentManager:
    def __init__(self, config_manager: ConfigManager):
        self.config_mgr = config_manager
        self.digestor: Optional[Digestor] = None
        self.embedder: Optional[Embedder] = None
        self.available_models: List[str] = []
        self.reload_components()
    def reload_components(self) -> None:
        logging.info("ComponentManager: Reloading components/models...")
        config = self.config_mgr.get_config()
        self.digestor = None
        self.embedder = None
        try: self.digestor=Digestor(config); logging.info("Digestor initialized OK.")
        except Exception as e: logging.exception(f"Digestor init FAILED: {e}")
        try: self.embedder=Embedder(config); logging.info("Embedder initialized OK.") if self.embedder and self.embedder.is_available() else logging.warning("Embedder init OK but UNAVAILABLE.")
        except Exception as e: logging.exception(f"Embedder init FAILED: {e}")
        self._fetch_models_from_backend()

    def _derive_base_url(self, full_url: Optional[str]) -> Optional[str]:
        # (Remains the same)
        if not full_url: return None
        try: parsed=urlparse(full_url); return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
        except Exception: return None


    def _fetch_models_from_backend(self) -> None:
        """Fetches models list. Only adds Auth header if api_key has value."""
        config = self.config_mgr.get_config()
        api_type = config.get("api_type", "Custom")
        backend_url = config.get("backend_url")
        api_key = config.get("api_key", "") # Get the maybe-present key
        preset = self.config_mgr.get_api_preset_for_type(api_type)

        logging.info(f"Attempting model fetch for API type '{api_type}'...")

        if not preset or not preset.get("list_endpoint") or not backend_url:
            logging.warning(f"Model fetch ABORTED -> No list endpoint for '{api_type}' or URL missing.")
            self.available_models = []; return

        base_url = self._derive_base_url(backend_url)
        if not base_url:
             logging.error(f"Model fetch FAIL -> Cannot derive BASE URL from: '{backend_url}'."); self.available_models=[]; return

        list_url = urljoin(base_url, preset["list_endpoint"])
        model_name_key = preset.get("name_key")
        if not model_name_key:
             logging.error(f"PRESET issue for '{api_type}': Missing 'name_key'."); return

        headers = {"Accept": "application/json"}
        # --- Updated Auth Header Logic ---
        if api_key: # Only add header if api_key string is not empty
            headers["Authorization"] = f"Bearer {api_key}"
            logging.debug(f"API Key found, adding Authorization header for {api_type}.")
        # --- End Update ---
        # (We can still optionally warn if a preset *expects* a key but none was provided)
        elif preset.get("needs_key"):
            logging.warning(f"API Type '{api_type}' preset indicates a key is usually needed, but none was provided in config.")


        logging.debug(f"Querying Models: URL='{list_url}', NameKey='{model_name_key}', KeyProvided={bool(api_key)}")
        fetched_model_names = []
        try:
            response = requests.get(list_url, headers=headers, timeout=20); response.raise_for_status(); data = response.json()
            models_data_list = data.get("models" if api_type == "Ollama" else "data", [])
            if isinstance(models_data_list, list):
                fetched_model_names = [m.get(model_name_key) for m in models_data_list if isinstance(m, dict) and m.get(model_name_key)]
                self.available_models = sorted(list(set(filter(None, fetched_model_names))));
                logging.info(f"Model fetch SUCCESS -> Found {len(self.available_models)} Models for '{api_type}'.")
            else: logging.error(f"PARSE Models RESPONSE ERROR -> Expected LIST, Got {type(models_data_list)}! Type='{api_type}'."); self.available_models = []
        # --- Exception Handling (remain same) ---
        except requests.exceptions.Timeout: logging.error(f"TIMEOUT Fetching models {list_url}")
        except requests.exceptions.ConnectionError: logging.error(f"CONNECTION Failed {list_url}")
        except requests.exceptions.RequestException as req_e:
             rc = req_e.response.status_code if req_e.response else '??'; body = req_e.response.text[:100] if req_e.response else ''
             loglvl = logging.CRITICAL if rc == 401 else logging.ERROR
             logging.log(loglvl, f"HTTP Error fetch models ({rc}) - URL:{list_url} Body=[{body}...]")
             if rc==401: logging.error("--> Hint: CHECK API KEY configuration validity! <--")
        except Exception: self.available_models = []; logging.exception(f"UNEXPECTED Err Fetching models: {list_url}")

    # --- Public Accessors / Readiness Checks ---
    def get_digestor(self) -> Optional[Digestor]: return self.digestor
    def get_embedder(self) -> Optional[Embedder]: return self.embedder
    def get_available_models(self) -> List[str]: return self.available_models.copy()
    @property
    def is_digestor_ready(self) -> bool: return self.digestor is not None
    @property
    def is_embedder_ready(self) -> bool: return self.embedder is not None and self.embedder.is_available()