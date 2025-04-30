import logging
import requests # Needed only for model fetching
from urllib.parse import urlparse, urljoin
from typing import Optional, List, Dict, Any
from .config_manager import ConfigManager, API_PRESETS # Import config dependency

# Import actual component classes (use TYPE_CHECKING for safety against cycles if needed)
from ..digestor import Digestor
from ..embedder import Embedder


class ComponentManager:
    """Initializes, holds, and manages core components (Digestor, Embedder)
    and fetches available models based on configuration."""

    def __init__(self, config_manager: ConfigManager):
        self.config_mgr = config_manager
        self.digestor: Optional[Digestor] = None
        self.embedder: Optional[Embedder] = None
        self.available_models: List[str] = []
        self.reload_components() # Initial population

    def reload_components(self) -> None:
        """(Re)Initializes components based on the *current* config from ConfigManager."""
        logging.info("ComponentManager: Reloading core components and available models...")
        config = self.config_mgr.get_config() # Get latest full config

        # --- Reset ---
        self.digestor = None
        self.embedder = None
        # Keep available_models cache unless fetch succeeds

        # --- Initialize Digestor ---
        if not config.get('backend_url') or not config.get('model_name'):
             logging.warning("Digestor disabled - missing 'backend_url' or 'model_name' in config.")
        else:
            try:
                self.digestor = Digestor(config)
                if self.digestor: logging.info("Digestor initialized OK.")
                else: raise RuntimeError("Digestor() classreturned None unexpectedly!?")# Should normally raise error IF problem, not Return None.. ok check.
            except Exception as e:
                logging.exception(f"Digestor initialization FAILED critically: {e}")

        # --- Initialize Embedder ---
        # Assumes 'api' type embedder using base config.
        try:
            self.embedder = Embedder(config) # Embedder reads required fields from config
            if self.embedder and self.embedder.is_available():
                 logging.info("Embedder initialized OK and API endpoint seems ready.")
            elif self.embedder:
                 logging.warning("Embedder init OK, but reported UNAVAILABLE (likely base URL issue). Embeddings will be skipped.")
            else: raise RuntimeError("Embedder() call returned None?")
        except Exception as e:
            logging.exception(f"Embedder initialization FAILED critically: {e}")

        # --- Fetch Models ---
        self._fetch_models_from_backend()


    def _derive_base_url(self, full_url: Optional[str]) -> Optional[str]:
        """Safely gets scheme://netloc part of a URL."""
        if not full_url: return None
        try:
            parsed = urlparse(full_url)
            return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else None
        except Exception: return None


    def _fetch_models_from_backend(self) -> None:
        """Fetches models list based on current config. Updates self.available_models."""
        # Fetch latest config values inside the fetching function ensures using current state. ok.
        config = self.config_mgr.get_config()
        api_type = config.get("api_type", "Custom")
        backend_url = config.get("backend_url")  # Base url calculation still needed
        api_key = config.get("api_key", "")
        preset = self.config_mgr.get_api_preset_for_type(api_type) # Gets Preset Dict using Helper for safe access..

        logging.info(f"Attempting model fetch for API type '{api_type}'...")

        if not preset or not preset.get("list_endpoint") or not backend_url:
            logging.warning(f"Model fetching ABORTED -> No list endpoint configured for type '{api_type}' or backend URL missing.")
            # Should we clear model list here?? Or leave STALE list...? --> CLEARING Seems safer. User knows NO models Fetched this run.. OK.
            self.available_models = []
            return

        base_url = self._derive_base_url(backend_url) # Deriving BASE URL Needed..! FROM the MAIN config URL..
        if not base_url:
             logging.error(f"Model fetch FAIL -> Cannot Derive BASE valid URL From:'{backend_url}'. Check Config Format.")
             self.available_models = [] # CLEAR cache --> Bad URL.
             return

        # Construct FINAL Endpoint URL Correctly.. use BASE inferred + API endpoint standard path / definition ..
        list_endpoint_path = preset["list_endpoint"]   # Contains endpoint like --> "/api/tags" .. OR "/v1/models" .. etc.. OK.
        list_url = urljoin(base_url, list_endpoint_path) # >>---- Correctly JOIN base + endpoint Path! ----<<

        model_name_key = preset.get("name_key") # Key like 'name' or 'id' needed for parsing OK safe GET.
        if not model_name_key:
             logging.error(f"CONFIG PRESET issue for Api -> '{api_type}' --> Missing the 'name_key' definition! Cannot parse. Fix Presets map.") ; return # BAD preset setup.. FATAL config ERR path.

        headers = {"Accept": "application/json"}
        # Add API key if preset Requires AND it exists.
        if preset.get("needs_key"):
            if api_key: headers["Authorization"] = f"Bearer {api_key}"
            else: logging.error(f"Cannot fetch Models 'AUTH KEY NEEDED' by '{api_type}' preset BUT NONE Configured."); self.available_models=[]; return; # AUTH required MISSING Key -> ABORT Fetch.


        logging.debug(f"Querying Models: URL='{list_url}', NameKey='{model_name_key}', NeedsKey={preset.get('needs_key')}")
        fetched_model_names= [] # Temp store for models fetched OK.
        try:
            # PERFORM Actual request TO List / Detect Models from backend...>
            response = requests.get(list_url, headers=headers, timeout=20) # Use GENEROUS timeout model lists can be LARGE > 10s+?? 20 ok..
            response.raise_for_status() # Check for HTTP ERRORS (4xx/5xx..)
            data = response.json() # Parse response IF Status OK...

            # Parse based on API Type Structure... ( Needs Preset Correct definitions)
            models_data_list = [] # Default EMPTY list.
            if api_type == "Ollama" : # Ollama response shape unique--> { Models : [{ name: ...} ] .. } -> Get LIST from data dict..
                 models_data_list = data.get("models", []) # Safe ACCESS with Default Empty list if Key Miss.
            else: # Standard OpenAI style endpoint shape -> { data : [ { ID : ...}, {...} ] } ...
                 models_data_list = data.get("data", [])  # Safe get with defaults.. ok..

            # PROCESS the LIST of model dictionaries (or Whatever backend API returned structure was...)
            if isinstance(models_data_list, list) :
                # --> Extract the NAME FIELD from Each item in list using correct preset Key.. OK.. List Comprehension NICE.
                fetched_model_names = [m.get(model_name_key) for m in models_data_list if isinstance(m, dict) and m.get(model_name_key)]
                # -->>>> SANATIZE Results --> Ensure Unique Names + Sorted Alpha etc... <<-- NICE CLEANUP.
                self.available_models = sorted(list(set(filter(None, fetched_model_names)))); # Filter Nones + Unique Set + Sort.. DONE.
                logging.info(f"Model fetch SUCCESS -> Found {len(self.available_models)} Models For type '{api_type}'.")

            else: # PARSE FAILED -> Expect List data -> Got something Else?? Backend bad json shape? ...
                 logging.error(f"Parse Models RESPONSE ERROR -> Expected LIST structure under key 'models' or 'data'-> But Got TYPE: '{type(models_data_list)}'! Type='{api_type}'. FIX PRESET or Check BACKEND API output FORMAT!.")
                 self.available_models= [] # CLEAR on PARSE Failure... Signal User Needs INVESTIGATION...

        # --- EXCEPTION Handling during model fetch -->>
        except requests.exceptions.Timeout: logging.error(f"TIMEOUT Fetching models {list_url} -> Check backend service!")
        except requests.exceptions.ConnectionError: logging.error(f"CONNECTION Failed {list_url} -> Backend OFFLINE?")
        except requests.exceptions.RequestException as req_e:
             rc = req_e.response.status_code if req_e.response else '??' ; body = req_e.response.text[:100] if req_e.response else ''
             loglvl=logging.CRITICAL if rc==401 else logging.ERROR # Elevate log LVL for AUTH errors.. Helps debug user setup..
             logging.log(loglvl, f"HTTP Error fetch models ({rc}) - URL:{list_url} Body=[{body}...] ");# Log Err BODY hint useful debug..
             if rc==401: logging.error("--> Hint: CHECK Your API KEY validity / configuration for this Provider.<---")
             # NOTE: KEEPING old `available_models` cache on HTTP error path seems reasonable... UI shows last GOOD list maybe better than Empty? Yes.
        except Exception: # Catch ALL others.. unexpected json bugs / local variable issues etc..
             self.available_models = [] # CLEAR on unexpected parse error.. maybe safest..
             logging.exception(f"UNEXPECTED Err Fetching models data : {list_url}")
        # End Exception handling for fetch... Function finished...`self.available_models` state updated OR kept old..


    # --- Public Accessors / Readiness Checks ---
    def get_digestor(self) -> Optional[Digestor]: return self.digestor
    def get_embedder(self) -> Optional[Embedder]: return self.embedder
    def get_available_models(self) -> List[str]: return self.available_models.copy() # Return COPY list users cannot mutate internal state.

    @property
    def is_digestor_ready(self) -> bool: return self.digestor is not None
    @property
    def is_embedder_ready(self) -> bool: return self.embedder is not None and self.embedder.is_available()