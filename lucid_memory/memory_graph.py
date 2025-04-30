import json
from typing import Dict, List, Optional
from lucid_memory.memory_node import MemoryNode
import logging # Added logging
import os # Added OS

class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}

    def add_node(self, node: MemoryNode):
        """Adds or updates a node in the graph."""
        if not node or not getattr(node, 'id', None):
             logging.warning("MemoryGraph received invalid/empty node, skipping add.")
             return
        if node.id in self.nodes:
            logging.debug(f"MemoryGraph: Updating existing node {node.id}")
            # Optionally merge or handle updates intelligently later?
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id)

    def search_by_tag(self, tag: str) -> List[MemoryNode]:
        """Searches for nodes containing a specific tag (Case-insensitive)."""
        matches = []
        tag_lower = tag.lower() # Perform lower once
        for node in self.nodes.values():
            if node.tags and any(t.lower() == tag_lower for t in node.tags):
                matches.append(node)
        return matches

    def save_to_json(self, filepath: str):
        """Enhanced save with error handling and backup possibility."""
        temp_filepath = filepath + ".tmp" # Use temporary file for atomicity
        backup_filepath = filepath + ".bak" # Path for backing up old file

        logging.info(f"MemoryGraph: Attempting save to {filepath} ({len(self.nodes)} nodes)...")
        try:
            # Serialize current graph to dict
            graph_dict = {node_id: node.to_dict() for node_id, node in self.nodes.items()}

            # Check if serialization produced something runnable
            if not isinstance(graph_dict, dict):
                 raise TypeError(f"Serialization failed? Not Dict type: {type(graph_dict)}")

            # Writing to temporary file first
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(temp_filepath, 'w', encoding='utf-8') as f:
                json.dump(graph_dict, f, indent=2,ensure_ascii=False) # ensure_ascii=False good

            # If temp write successful, manage backup and replace
            if os.path.exists(filepath): # Does original exist?
                try:
                   if os.path.exists(backup_filepath): os.remove(backup_filepath) # Delete old backup
                   os.rename(filepath, backup_filepath) # Backup current file
                   logging.debug(f"Backed up existing {filepath} to {backup_filepath}")
                except OSError as e:
                    logging.warning(f"Couldn't remove old backup or backup {filepath}: {e} - Continuing...")
            os.rename(temp_filepath, filepath)
            logging.info(f"MemoryGraph: Successfully saved {len(self.nodes)} nodes to {filepath}.")

        except Exception as e:
             logging.error(f"MemoryGraph: Save FAILED for {filepath}: {e}", exc_info=True)
             # If temp failed, CLEANUP TEMPORARY FILE
             if os.path.exists(temp_filepath):
                 try: os.remove(temp_filepath); logging.info(f"Cleaned up failed temporary file {temp_filepath}")
                 except Exception as erem: logging.warning(f"Failed cleanup of {temp_filepath}. {erem}")

             # Propagate the error for the Processor to know save failed
             raise # Re raise underlying error after logging

    def load_from_json(self, filepath: str):
        """Loads graph data from json WITH backup handling mechanism!"""
        logging.info(f"MemoryGraph: Attempting to load graph from {filepath}...")
        if not os.path.exists(filepath):
             logging.warning(f"Primary file {filepath} not found. Trying backup...")
             backup_filepath = filepath + ".bak"
             if os.path.exists(backup_filepath):
                  logging.info(f"Found backup {backup_filepath}, attempting load from backup.")
                  filepath = str(os.path.realpath(backup_filepath))# update the path being used
             else:
                  logging.error(f"No primary or backup memory file found. Initializing empty graph")
                  self.nodes = {} # Ensure it remains empty
                  return # Nothing else to run
        try:            
            with open(filepath, 'r', encoding='utf-8') as f:
                 data = json.load(f)

            new_nodes: Dict[str, MemoryNode] = {} # Build map in temporary variable first
            loaded_count = 0
            error_count = 0
            # Ensure loading uses the updated .from_dict that handles potential bad embeddings etc. better
            for node_id, node_dict in data.items():
                try:
                     # Provide default ID if missing inside dictionary when loading
                     if 'id' not in  node_dict : node_dict['id'] = node_id
                     # Skip dictionary that doesn't appear to be a node structure
                     if not isinstance(node_dict, dict) or 'summary' not in node_dict: raise TypeError("Dict looks incomplete")

                     new_nodes[node_id] = MemoryNode.from_dict(node_dict)
                     loaded_count += 1
                except (TypeError, ValueError) as e:
                    logging.warning(f"Corrupt node or conversion error deserializing node '{node_id}': {e}. Skipping.", exc_info=False)# Don't traceback for common errors maybe
                    error_count += 1

            self.nodes = new_nodes# Replace our internal dictionary with the loaded one
            log_message=f"MemoryGraph: Load OK ({loaded_count} nodes from {filepath})."
            if error_count > 0 : log_message += f" ENCOUNTERED {error_count} load errors (Potentially bad node formats skipped)."
            logging.info(log_message)

        except (json.JSONDecodeError) as json_err: # Handle catastrophic file corruptions
            logging.critical(f"CRITICAL: JSON Parse Error reading {filepath}: ({json_err}). Cannot load graph.", exc_info=True)
            # Choice: Keep current state? Nuke everything? Let's NUKE it assuming user will try fresh process
            self.nodes = {}
            raise # Propagate for controller etc
        except Exception as e:
            logging.error(f"Unexpected error loading graph from {filepath}: {e}", exc_info=True)
            # Nuke for safety??
            self.nodes = {}
            raise