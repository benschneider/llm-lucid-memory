# Tests theDigestor -> Processor -> Embedder -> MemoryNode update pipeline
import sys
import os
import json
import logging
import time
from typing import List, Dict, Any, Optional, Callable

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# logging.info(f"Adding Proj Root : {project_root} to python sys path.") # MODIFIED: Use logging

try:  # Import Modules... Handle if Fails Nicely..
    from lucid_memory.chunker import chunk_file
    from lucid_memory.digestor import Digestor
    from lucid_memory.embedder import Embedder
    from lucid_memory.processor import ChunkProcessor
    from lucid_memory.memory_graph import MemoryGraph
    from lucid_memory.memory_node import MemoryNode
    from lucid_memory.config_manager import ConfigManager # MODIFIED: Use ConfigManager
except ImportError as e: # MODIFIED: Corrected typo
    logging.exception(f"IMPORT FAIL: Check install/paths `pip install -e . ` etc! Error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(threadName)s - %(module)s - %(message)s')
TEST_MEMORY_FILE = os.path.join(project_root, "tests", "test_pipeline_memory_graph.json") # MODIFIED: Path
TEST_CONFIG_FILE = os.path.join(project_root, "tests", "test_temp_config.json") # MODIFIED: Path

# MODIFIED: Added dummy Python content
PY_CONTENT = """
def hello_world(name):
    '''Greets the person'''
    print(f"Hello, {name}!")
    return f"Greetings sent to {name}"

class MyClass:
    def __init__(self, value):
        self.value = value

    def get_value(self):
        return self.value
"""
DEFAULT_EMBEDDING_API_MODEL = "nomic-embed-text" # MODIFIED: Defined default

#---------------------CONFIG CREATION Modified--------------------
def create_test_config(test_embedding_model: Optional[str] = DEFAULT_EMBEDDING_API_MODEL) -> bool: # MODIFIED: Return bool
    default_chat_llm_name = "mistral"
    tested_embedding_model = test_embedding_model if test_embedding_model and test_embedding_model.strip() else DEFAULT_EMBEDDING_API_MODEL

    conf_data_dict = {
        "api_type": "Ollama", # ADDED for completeness
        "backend_url": "http://localhost:11434/v1/chat/completions",
        "model_name": default_chat_llm_name,
        "embedding_api_model_name": tested_embedding_model, # MODIFIED: Key used by Embedder
        "local_proxy_port": 8002, # MODIFIED: Different port for test
        "api_key": "" # ADDED for completeness
    }

    try:
        os.makedirs(os.path.dirname(TEST_CONFIG_FILE), exist_ok=True)
        with open(TEST_CONFIG_FILE, "w") as cfg_f:
            json.dump(conf_data_dict, cfg_f, indent=2)
        logging.info(f"TEMP TEST CONFIG CREATED ('{TEST_CONFIG_FILE}'). Using CHAT='{default_chat_llm_name}' and EMBED='{tested_embedding_model}'")
        
        # MODIFIED: Check for prompts.yaml relative to lucid_memory package
        prompts_path = os.path.join(project_root, "lucid_memory", "prompts.yaml")
        if not os.path.exists(prompts_path):
            logging.error(f"*** CRITICAL MISSING '{prompts_path}'")
            # Attempt to create a dummy prompts.yaml for the test to proceed
            try:
                dummy_prompts = {
                    'summary': 'Summarize: {raw_text}',
                    'key_concepts': 'Concepts for: {raw_text}',
                    'tags': 'Tags for: {raw_text}',
                    'questions': 'Questions for: {raw_text}',
                    'code_dependencies': 'Dependencies for code: {code_chunk}',
                    'code_outputs': 'Outputs for code: {code_chunk} given dependencies: {dependency_list}'
                }
                with open(prompts_path, "w") as pf:
                    import yaml
                    yaml.dump(dummy_prompts, pf)
                logging.warning(f"Created DUMMY prompts.yaml at {prompts_path} for testing.")
            except Exception as e_prompts:
                logging.error(f"Failed to create dummy prompts.yaml: {e_prompts}")
                return False
        return True
    except (IOError, PermissionError) as e: # MODIFIED: Capture exception
        logging.exception(f"TEMP CONFIG ERR Write Error Permission maybe? STOPP: {e}")
        return False

#---------------TEST ClEANUP Func-------------------
def cleanup_test_files():
    if os.path.exists(TEST_MEMORY_FILE):
        try: os.remove(TEST_MEMORY_FILE); logging.info("Removed temp JSON graph Ok ")
        except OSError as e: logging.warning(f"Could not remove {TEST_MEMORY_FILE}: {e}")
    if os.path.exists(TEST_CONFIG_FILE):
        try: os.remove(TEST_CONFIG_FILE); logging.info("Removed temp JSON configuration Ok ")
        except OSError as e: logging.warning(f"Could not remove {TEST_CONFIG_FILE}: {e}")

#-------------MAIN FUNCTION Execution-----------------
def main():
    status_msgs = []
    completion_info = {}

    def status_cb(message: str, current_step: Optional[int]=None, total_steps: Optional[int]=None, detail: Optional[str]=None): # MODIFIED: Signature match
        log_msg = f"STATUS CB: {message}"
        if current_step is not None and total_steps is not None: log_msg += f" ({current_step}/{total_steps})"
        if detail: log_msg += f" - {detail}"
        logging.info(log_msg)
        status_msgs.append(message)

    def final_cb(changed: bool):
        logging.info(f"FINAL CB: graph changed={changed}")
        completion_info['graph_change'] = changed

    if not create_test_config():
        logging.critical("TEST SETUP Err EXIT NOW ")
        sys.exit(1)

    config_obj_read = None
    try:
        with open(TEST_CONFIG_FILE, "r") as cfg_r:
            config_obj_read = json.load(cfg_r)
        logging.debug(f"LOADING config For TEST: {config_obj_read}")
    except Exception as e: # MODIFIED: Corrected typo
        logging.exception(f"CANNOT Load Temp Config NEEDED!! {e}")
        cleanup_test_files() # MODIFIED: Cleanup on failure
        sys.exit(1)

    digestor = None
    embedder = None
    try:
        digestor = Digestor(config_obj_read)
        embedder = Embedder(config_obj_read)
        
        final_embed_model_requested = "FAILED_INIT_EMB"
        if embedder and embedder.is_available(): # MODIFIED: Check embedder instance
             final_embed_model_requested = embedder.api_model_name # MODIFIED: Access corrected attribute
        logging.info(f"----- Digestor Init = {digestor is not None} ---- CHECK FOR PROMPT ERRORS maybe?")
        logging.info(f"-----Embedder Init = {embedder.is_available() if embedder else False} Targeting FINAL model {final_embed_model_requested} --- CHECK LOGS ")
    except Exception as e: # MODIFIED: Corrected typo
        logging.exception(f"INIT components FAILED STOP!! {e}")
        cleanup_test_files()
        sys.exit(1)

    graph = MemoryGraph()
    # MODIFIED: Set path directly in ChunkProcessor or rely on its default if suitable
    # ChunkProcessor.MEMORY_GRAPH_PATH = TEST_MEMORY_FILE # This might be better passed or configured

    processed_ok = True
    try:
        # Use chunk_file which expects a path, or adapt to use `chunk` with content
        # For this test, let's simulate chunk_file by writing PY_CONTENT to a temp file
        temp_py_file = os.path.join(os.path.dirname(TEST_CONFIG_FILE), "dummy_fileForTEST.py")
        with open(temp_py_file, "w") as f_py:
            f_py.write(PY_CONTENT)

        chunks = chunk_file(temp_py_file, PY_CONTENT) # Pass content to avoid re-reading
        os.remove(temp_py_file) # Clean up temp file

        if not chunks:
            logging.error("**** CHUNKER BROKEN?? No chunks.. FIX NEEDED") # MODIFIED: Use logging
            cleanup_test_files()
            exit(1)

        logging.info(f"OK GOT {len(chunks)} Chunks... Creating PROCESSOR....")
        # MODIFIED: Ensure ChunkProcessor uses the test memory file path
        processor = ChunkProcessor(
            digestor=digestor, 
            embedder=embedder, 
            memory_graph=graph,
            status_callback=status_cb, 
            completion_callback=final_cb,
            memory_graph_path_override=TEST_MEMORY_FILE # MODIFIED: Pass override path
        )

        start_time = time.monotonic()
        processor.process_chunks(chunks, original_filename="dummy_script.py")
        end_time = time.monotonic()
        duration = end_time - start_time
        logging.info(f"======== OK Processor finished task.. Time Taken: {duration:.2f} secs =========")

        logging.info("Validating MemoryGraph State And Node Embeddings ....")
        # MODIFIED: Load graph from the test file to check persistence
        final_graph = MemoryGraph()
        final_graph.load_from_json(TEST_MEMORY_FILE)

        graph_nodes = list(final_graph.nodes.values())
        node_count = len(graph_nodes)

        assert node_count > 0, f"FAILURE: NO Nodes created AT ALL in Graph?? {node_count} Found. Chunker OR processor FAIL?"
        logging.info(f"PASS=> Created {node_count} Graph Nodes total SUCCESS!")

        found_embeddings = 0
        # MODIFIED: Embedder availability check
        embedder_was_available = embedder.is_available() if embedder else False

        for node_idx, node in enumerate(graph_nodes):
            embData = getattr(node, 'embedding', None) # MODIFIED: Access embedding attribute if it exists
            if embData:
                assert isinstance(embData, list), f"FAIL Node {node.id} EMBEDDING NOT List (type: {type(embData)})???"
                if not embData: # MODIFIED: Check for empty list
                    logging.warning(f"WARN: Node {node.id} Embed List Empty {embData}?")
                    continue
                if not all(isinstance(x, (float, int)) for x in embData): # MODIFIED: Corrected loop variable X
                    logging.warning(f"WARN: Node {node.id} Got non NUM in emb{embData}")
                    continue # Potentially skip counting this if strict
                found_embeddings += 1 # MODIFIED: Corrected increment

        logging.info(f"FINAL CHECK => Found {found_embeddings} Embedded nodes. Embedder READY was={embedder_was_available}")

        if embedder_was_available:
            assert found_embeddings == node_count, \
                f"FAILURE: Embedder WAS Ready -> Expected ALL {node_count} Nodes embedded BUT ONLY {found_embeddings} GOT Them!"
        else:
            assert found_embeddings == 0, \
                f"FAILURE: Embedder WAS NOT READY But found {found_embeddings} EMBEDDED? State/Logic ERR!"
        logging.info("======= EMBEDDING PIPELINE TEST PASSED =======")

    except Exception as e:
        processed_ok = False
        logging.exception(f"MAIN TEST EXECUTION FAILED: {e}")
    finally:
        cleanup_test_files()
        if not processed_ok:
            logging.error("======= EMBEDDING PIPELINE TEST FAILED =======")
            sys.exit(1)

if __name__ == "__main__":
    main()