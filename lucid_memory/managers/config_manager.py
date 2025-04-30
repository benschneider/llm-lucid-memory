# lucid_memory/managers/config_manager.py
import os
import json
import logging
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ConfigManager - %(message)s')

# Define paths relative to this file's location if needed elsewhere securely
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PACKAGE_DIR = os.path.dirname(CONFIG_DIR) # Up one level to lucid_memory/
CONFIG_FILE_PATH = os.path.join(MAIN_PACKAGE_DIR, "proxy_config.json") # Correct path relative to package

# Define DEFAULTS and PRESETS here for encapsulation
DEFAULT_CONFIG = {
    "api_type": "Ollama",
    "backend_url": "http://localhost:11434/v1/chat/completions",
    "model_name": "mistral",
    "local_proxy_port": 8000,
    "api_key": ""
}
API_PRESETS = { # Keep presets defined alongside config logic
    "Ollama": {"url": "http://localhost:11434/v1/chat/completions", "needs_key": False, "list_endpoint": "/api/tags", "name_key": "name"},
    "LM Studio": {"url": "http://localhost:1234/v1/chat/completions", "needs_key": False, "list_endpoint": "/v1/models", "name_key": "id"},
    "OpenAI": {"url": "https://api.openai.com/v1/chat/completions", "needs_key": True, "list_endpoint": "/v1/models", "name_key": "id"},
    "OpenRouter": {"url": "https://openrouter.ai/api/v1/chat/completions", "needs_key": True, "list_endpoint": "/v1/models", "name_key": "id"},
    "Custom": {"url": "", "needs_key": False, "list_endpoint": None, "name_key": None}
}


class ConfigManager:
    """Handles loading, validation, saving, and accessing application configuration."""
    def __init__(self, config_path: str = CONFIG_FILE_PATH):
        self.config_path = config_path
        self._config: Dict[str, Any] = self._load_config() # Load on instantiation

    def _load_config(self) -> Dict[str, Any]:
        """Loads JSON config, merges over defaults. Handles errors."""
        cfg = DEFAULT_CONFIG.copy() # Start with base defaults
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    cfg.update(loaded) # Apply user settings
                    logging.info(f"Config loaded OK from: '{self.config_path}'")
                else:
                   logging.error(f"Config file '{self.config_path}' is not a valid JSON dictionary. Using defaults.")
            except json.JSONDecodeError:
                 logging.exception(f"Config file '{self.config_path}' JSON parse error. Using defaults. Fix file.")
            except IOError:
                logging.exception(f"Config file '{self.config_path}' READ error. Using defaults.")
            except Exception:
                 logging.exception(f"Unexpected error loading config '{self.config_path}'. Using defaults.")
        else:
            logging.warning(f"Config file '{self.config_path}' not found. Initializing with defaults and saving.")
            # Attempt to save defaults only if file doesn't exist
            try:
                self._save_config(cfg) # Save the default state immediately
            except Exception as e_save:
                 logging.error(f"FAILED to save initial default config! Check permissions/path. Err: {e_save}")
        return cfg

    def _save_config(self, config_dict: Dict[str, Any]) -> bool:
        """Saves the provided configuration dictionary to the config file."""
         # Ensure SAVED version reflects TOTAL config space -> Start with Defaults + apply provided Overrides...
        final_config = DEFAULT_CONFIG.copy()
        final_config.update(config_dict) # Update defaults with incoming keys/values

        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as wf:
                 json.dump(final_config, wf, indent=2)
            logging.info(f"Configuration successfully saved to '{self.config_path}'")
            return True
        except (IOError, TypeError, ValueError) as save_err:
            logging.error(f"Config SAVE FAILED -> '{self.config_path}'. Data/Permission issue? ({type(save_err).__name__}): {save_err}", exc_info=True)
            return False

    def get_config(self) -> Dict[str, Any]:
        """Returns a COPY of the current complete configuration."""
        full_cfg = DEFAULT_CONFIG.copy()
        full_cfg.update(self._config) # Apply runtime state over base defaults
        return full_cfg

    def get_value(self, key: str, default: Any = None) -> Any:
         """Safely gets a specific config value, falling back to stored default."""
         # Access internal state, fallback to class default Map, then final fallback provided by user...
         return self._config.get(key, DEFAULT_CONFIG.get(key, default))


    def update_config(self, partial_update: Dict[str, Any]) -> bool:
        """
        Updates runtime config with new values, validates, and persists.
        Returns True if successful, False otherwise.
        """
        logging.debug(f"ConfigManager: Attempting update with: {partial_update}")

        # --- Pre-Save Validation ---
        temp_merged_config = self.get_config() # Get full current state COPY
        temp_merged_config.update(partial_update) # Simulate merge

        # Validate Merged State (ensure required keys exist and have non-empty values)
        required_keys = ['backend_url', 'model_name', 'api_type']
        missing_or_empty = [k for k in required_keys if not str(temp_merged_config.get(k, "")).strip()]
        if missing_or_empty:
            logging.error(f"Config Update REJECTED - Missing/Empty required fields after merge: {missing_or_empty}")
            return False

        # Port Validation
        port_str = str(temp_merged_config.get('local_proxy_port', '8000')).strip()
        if not port_str.isdigit() or not (1024 <= int(port_str) <= 65535):
            logging.error(f"Config Update REJECTED - Invalid Port Value: '{port_str}'. Need 1024-65535.")
            return False

        # --- Validation Passed - Apply & Persist ---
        logging.info("Config validation passed. Applying updates and saving...")
        self._config.update(partial_update) # Update IN-MEMORY state

        if self._save_config(self._config): # Save the NOW UPDATED in-memory state
             logging.info("Config state saved to file successfully.")
             return True
        else:
             # Save failed! Error already logged by _save_config
             logging.critical("Config update applied in-memory BUT SAVE FAILED! State mismatch!")
             # Should we revert in-memory state? Maybe not necessary if next load fixes it? Risk...
             # For now, just report failure.
             return False

    # Optionally add methods to specifically get API presets etc.
    def get_api_presets(self) -> Dict[str, Dict]:
         return API_PRESETS.copy() # Return copy

    def get_api_preset_for_type(self, api_type: Optional[str]) -> Optional[Dict]:
         """ Gets preset details based on string name, handles None. """
         if not api_type: return API_PRESETS.get("Custom") # Default fallback if type unset / None
         return API_PRESETS.get(api_type) # Return PRESET Map Dict or-> None if key bad...