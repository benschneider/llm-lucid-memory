import unittest
from unittest.mock import patch, MagicMock
import requests # For raising requests.exceptions

from lucid_memory.embedder import Embedder
from lucid_memory.memory_node import MemoryNode # Assuming MemoryNode is needed for context

# A default dummy config for the embedder for most tests
DUMMY_CONFIG = {
    "backend_url": "http://localhost:11434/v1/chat/completions", # Base URL part is used
    "embedding_api_model_name": "test-embedding-model",
    "api_key": "test_api_key_if_needed" # Embedder now checks for this
}

# A dummy config that would make the embedder not available
INVALID_CONFIG_NO_URL = {
    "embedding_api_model_name": "test-embedding-model",
}

class TestEmbedder(unittest.TestCase):

    def create_sample_node(self, node_id="test_node_1"):
        return MemoryNode(
            id=node_id,
            raw="This is raw text for testing.",
            summary="A test summary.",
            key_concepts=["testing", "embeddings"],
            tags=["test", "sample"],
            dependencies=["dep1"],
            produced_outputs=["out1"]
        )

    def test_embedder_initialization_valid_config(self):
        """Test that Embedder initializes correctly with a valid config."""
        embedder = Embedder(DUMMY_CONFIG)
        self.assertTrue(embedder.is_configured)
        self.assertTrue(embedder.is_available())
        self.assertEqual(embedder.api_model_name, "test-embedding-model")
        self.assertIn("/v1/embeddings", embedder.embedding_endpoint_url)
        self.assertEqual(embedder.api_key, "test_api_key_if_needed")

    def test_embedder_initialization_invalid_config(self):
        """Test that Embedder handles invalid config (e.g., missing URL)."""
        embedder = Embedder(INVALID_CONFIG_NO_URL)
        self.assertFalse(embedder.is_configured)
        self.assertFalse(embedder.is_available())

    def test_prepare_text_for_embedding(self):
        """Test the _prepare_text_for_embedding method."""
        embedder = Embedder(DUMMY_CONFIG)
        node = self.create_sample_node()
        prepared_text = embedder._prepare_text_for_embedding(node)

        self.assertIn("Summary: A test summary.", prepared_text)
        self.assertIn("Key Concepts/Logic: testing, embeddings", prepared_text)
        self.assertIn("Tags: test, sample", prepared_text)
        self.assertIn("Dependencies: dep1", prepared_text)
        self.assertIn("Outputs: out1", prepared_text)
        self.assertTrue(len(prepared_text) > 0)

    def test_prepare_text_for_embedding_minimal_node(self):
        embedder = Embedder(DUMMY_CONFIG)
        node = MemoryNode(id="min_node", raw="Raw only", summary="", key_concepts=[], tags=[])
        prepared_text = embedder._prepare_text_for_embedding(node)
        # It should fall back to raw content if it's short enough
        self.assertEqual(prepared_text, "Raw only")
    
    def test_prepare_text_for_embedding_empty_node(self):
        embedder = Embedder(DUMMY_CONFIG)
        node = MemoryNode(id="empty_node", raw="", summary="", key_concepts=[], tags=[])
        prepared_text = embedder._prepare_text_for_embedding(node)
        self.assertEqual(prepared_text, "")


    @patch('lucid_memory.embedder.requests.post')
    def test_generate_embedding_success(self, mock_post):
        """Test successful embedding generation with a mocked backend."""
        embedder = Embedder(DUMMY_CONFIG)
        sample_node = self.create_sample_node()
        dummy_vector = [0.1, 0.2, 0.3, 0.4, 0.5]

        # Configure the mock response from requests.post
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock() # Does nothing for success
        mock_response.json.return_value = {
            "data": [
                {"embedding": dummy_vector, "object": "embedding", "index": 0}
            ],
            "model": "test-embedding-model",
            "object": "list",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }
        mock_post.return_value = mock_response

        # Call the method to test
        embedding_result = embedder.generate_embedding(sample_node)

        # Assertions
        self.assertIsNotNone(embedding_result)
        self.assertEqual(embedding_result, dummy_vector)

        # Check that requests.post was called correctly
        expected_text = embedder._prepare_text_for_embedding(sample_node)
        expected_payload = {
            "input": expected_text,
            "model": "test-embedding-model"
        }
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], embedder.embedding_endpoint_url) # Check URL
        self.assertEqual(kwargs['json'], expected_payload)      # Check payload
        self.assertIn('Authorization', kwargs['headers'])       # Check API key in headers
        self.assertEqual(kwargs['headers']['Authorization'], f"Bearer {DUMMY_CONFIG['api_key']}")


    @patch('lucid_memory.embedder.requests.post')
    def test_generate_embedding_http_error(self, mock_post):
        """Test embedding generation when the backend returns an HTTP error."""
        embedder = Embedder(DUMMY_CONFIG)
        sample_node = self.create_sample_node()

        # Configure mock_post to raise an HTTPError
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        # Create an actual HTTPError instance to be raised
        http_error = requests.exceptions.HTTPError(response=mock_response)
        mock_post.side_effect = http_error # Raise the error when called
        # Or, more simply for raise_for_status:
        # mock_post.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError(response=mock_response)


        embedding_result = embedder.generate_embedding(sample_node)
        self.assertIsNone(embedding_result)

    @patch('lucid_memory.embedder.requests.post')
    def test_generate_embedding_timeout_error(self, mock_post):
        """Test embedding generation when the backend call times out."""
        embedder = Embedder(DUMMY_CONFIG)
        sample_node = self.create_sample_node()
        mock_post.side_effect = requests.exceptions.Timeout("Request timed out")

        embedding_result = embedder.generate_embedding(sample_node)
        self.assertIsNone(embedding_result)

    def test_generate_embedding_not_available(self):
        """Test that generate_embedding returns None if embedder is not available."""
        embedder = Embedder(INVALID_CONFIG_NO_URL) # Config that makes it unavailable
        sample_node = self.create_sample_node()
        
        self.assertFalse(embedder.is_available())
        embedding_result = embedder.generate_embedding(sample_node)
        self.assertIsNone(embedding_result)

    @patch('lucid_memory.embedder.requests.post')
    def test_generate_embedding_malformed_response(self, mock_post):
        """Test handling of malformed JSON response from embedding API."""
        embedder = Embedder(DUMMY_CONFIG)
        sample_node = self.create_sample_node()

        # Configure mock to return malformed JSON
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"error": "malformed_data"} # Missing "data" or "embedding"
        mock_post.return_value = mock_response

        embedding_result = embedder.generate_embedding(sample_node)
        self.assertIsNone(embedding_result)

if __name__ == '__main__':
    unittest.main()
Use code with caution.
Python
II. Potential Minor Adjustments in lucid_memory/embedder.py (Self-Correction/Refinement during mock testing):
After writing the tests, I can see a few areas in embedder.py that could be slightly more robust or align better with how the mock works.
Ensure API key is passed in generate_embedding if present. (It looks like this was already added: if self.api_key: headers['Authorization'] = f"Bearer {self.api_key}")
The _prepare_text_for_embedding fallback to node.raw is good. Let's ensure the test covers the case where raw is too long as well.
Let's assume lucid_memory/embedder.py is largely fine based on our previous refactoring, and the tests above will primarily target its interaction with requests.post. No direct changes to embedder.py are strictly required by the mock itself, but testing often reveals small improvements.
III. Adapting tests/test_embedding_pipeline.py to use a Mocked Embedder
Now, let's see how to use a similar mocking strategy within your existing pipeline test. The goal here is to have the ChunkProcessor use an Embedder whose network calls are mocked.
# In tests/test_embedding_pipeline.py

# Add at the top:
from unittest.mock import patch, MagicMock
# import requests # If you need to raise requests.exceptions for the mock

# ... (existing imports, PY_CONTENT, default model, config creation, cleanup) ...

# Modify the main() function:
def main():
    # ... (existing setup for status_cb, final_cb, create_test_config, load config_obj_read) ...

    # Mock requests.post specifically for the Embedder's generate_embedding method
    # This patch will be active for the duration of the 'with' block.
    dummy_vector_for_pipeline = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    # This mock will apply to ALL calls to requests.post. If Digestor also uses it,
    # it might also be mocked unless we get more specific with the patching target.
    # A more targeted patch would be 'lucid_memory.embedder.requests.post'
    with patch('lucid_memory.embedder.requests.post') as mock_embedder_requests_post:
        
        # Configure the mock for the embedder's calls
        mock_embedder_response = MagicMock()
        mock_embedder_response.raise_for_status = MagicMock()
        mock_embedder_response.json.return_value = {
            "data": [{"embedding": dummy_vector_for_pipeline}]
        }
        mock_embedder_requests_post.return_value = mock_embedder_response

        # --- Initialize Components (Digestor might still try real calls if not mocked) ---
        digestor = None
        embedder = None
        try:
            digestor = Digestor(config_obj_read) # Digestor might make real LLM calls
            embedder = Embedder(config_obj_read) # Embedder's HTTP calls will be mocked
            
            # ... (logging for component init) ...
            logging.info(f"----- Digestor Init = {digestor is not None} (Will attempt real LLM calls if backend is up) ----")
            logging.info(f"----- Embedder Init = {embedder.is_available() if embedder else False} (HTTP calls will be MOCKED) ---")

        except Exception as e:
            logging.exception(f"INIT components FAILED STOP!! {e}")
            cleanup_test_files()
            sys.exit(1)

        graph = MemoryGraph()
        
        processed_ok = True
        try:
            # ... (chunking logic remains the same) ...
            # temp_py_file = os.path.join(os.path.dirname(TEST_CONFIG_FILE), "dummy_fileForTEST.py")
            # with open(temp_py_file, "w") as f_py: f_py.write(PY_CONTENT)
            # chunks = chunk_file(temp_py_file, PY_CONTENT)
            # os.remove(temp_py_file)
            # if not chunks: ... exit ...

            # Create a dummy file for chunking if PY_CONTENT is simple
            # For more complex tests, you might have actual test files.
            if not os.path.exists(os.path.dirname(TEST_CONFIG_FILE)):
                os.makedirs(os.path.dirname(TEST_CONFIG_FILE)) # Ensure test dir exists
            temp_py_file_path = os.path.join(os.path.dirname(TEST_CONFIG_FILE), "test_pipeline_dummy_script.py")
            with open(temp_py_file_path, "w") as f:
                f.write(PY_CONTENT)
            
            chunks = chunk_file(temp_py_file_path, PY_CONTENT) # Pass content to chunk_file to avoid re-read
            # os.remove(temp_py_file_path) # Optional: remove after chunking for this test

            if not chunks:
                logging.error("**** CHUNKER BROKEN?? No chunks.. FIX NEEDED")
                cleanup_test_files()
                sys.exit(1)


            logging.info(f"OK GOT {len(chunks)} Chunks... Creating PROCESSOR....")
            processor = ChunkProcessor(
                digestor=digestor, 
                embedder=embedder, # This embedder will use the mocked requests.post
                memory_graph=graph,
                status_callback=status_cb, 
                completion_callback=final_cb,
                memory_graph_path_override=TEST_MEMORY_FILE
            )

            start_time = time.monotonic()
            processor.process_chunks(chunks, original_filename="dummy_script_for_pipeline.py")
            end_time = time.monotonic()
            # ... (logging for duration) ...

            # --- Validation ---
            final_graph = MemoryGraph()
            final_graph.load_from_json(TEST_MEMORY_FILE) # Load the graph saved by processor
            graph_nodes = list(final_graph.nodes.values())
            node_count = len(graph_nodes)

            assert node_count > 0, f"FAILURE: NO Nodes created AT ALL in Graph?? {node_count} Found."
            logging.info(f"PASS=> Created {node_count} Graph Nodes total SUCCESS!")

            found_embeddings = 0
            embedder_was_available_for_test = embedder.is_available() if embedder else False

            for node_idx, node in enumerate(graph_nodes):
                self.assertIsNotNone(node.summary, f"Node {node.id} summary is missing.") # Example basic check from Digestor
                
                embData = getattr(node, 'embedding', None)
                if embData:
                    self.assertIsInstance(embData, list, f"FAIL Node {node.id} EMBEDDING NOT List (type: {type(embData)})")
                    # Check if it's our dummy vector
                    self.assertEqual(embData, dummy_vector_for_pipeline, f"Node {node.id} embedding does not match mocked vector.")
                    if not embData: # Should not happen if dummy_vector is not empty
                        logging.warning(f"WARN: Node {node.id} Embed List Empty {embData}?")
                        continue 
                    if not all(isinstance(x, (float, int)) for x in embData):
                        logging.warning(f"WARN: Node {node.id} Got non NUM in emb{embData}")
                        continue
                    found_embeddings += 1

            logging.info(f"FINAL CHECK => Found {found_embeddings} Mock-Embedded nodes. Embedder READY during test was={embedder_was_available_for_test}")

            if embedder_was_available_for_test:
                # Every node should have received the dummy embedding
                self.assertEqual(found_embeddings, node_count,
                    f"FAILURE: Embedder WAS Ready & Mocked -> Expected ALL {node_count} Nodes to have the mock embedding, BUT ONLY {found_embeddings} GOT Them!")
            else: # Should not happen if config is set up for embedder to be available
                self.assertEqual(found_embeddings, 0,
                    f"FAILURE: Embedder WAS NOT READY But found {found_embeddings} EMBEDDED? State/Logic ERR!")
            
            logging.info("======= MOCKED EMBEDDING PIPELINE TEST PASSED (for embedding part) =======")
            # Note: Digestor might still fail if its LLM backend is down, this test only guarantees embedding part.

        except AssertionError: # Catch assertion errors from self.assert...
            processed_ok = False
            logging.exception("Pipeline Test Assertion FAILED.") # Logs traceback for assertion
        except Exception as e:
            processed_ok = False
            logging.exception(f"MAIN TEST EXECUTION FAILED: {e}")
        finally:
            cleanup_test_files()
            # Remove the dummy script created for chunking
            if 'temp_py_file_path' in locals() and os.path.exists(temp_py_file_path):
                try: os.remove(temp_py_file_path)
                except OSError: pass

            if not processed_ok:
                logging.error("======= EMBEDDING PIPELINE TEST FAILED (or an assertion failed) =======")
                sys.exit(1) # Ensure script exits with error code if test fails

# For running test_embedding_pipeline.py directly, it's not a unittest.TestCase,
# so self.assert... won't work. We should convert it to use standard assert or make it a TestCase.
# For now, I'll leave the self.assert... and you can run it as part of a unittest suite
# or refactor it to use plain `assert`.
# If running directly, replace self.assert... with plain assert.
# Example: self.assertEqual(a,b) -> assert a == b, f"{a} != {b}"

if __name__ == "__main__":
    # To make asserts work if run directly, we can temporarily wrap main in a dummy class
    # or just use plain assert statements as suggested above.
    # For simplicity of changes now, I'll assume it might be run via a test runner
    # that provides a `self` context or these will be converted.
    # If you run `python tests/test_embedding_pipeline.py`, the self.assert lines will fail.
    # You'd need to change them to `assert condition, message`.
    
    # Quick fix for direct run:
    class DummyTest(unittest.TestCase): # Create a dummy TestCase to allow self.assert...
        def run_main_as_test(self):
            # Capture standard asserts as instance methods
            TestEmbedderPipeline.assertEqual = self.assertEqual
            TestEmbedderPipeline.assertIsNone = self.assertIsNone
            TestEmbedderPipeline.assertIsNotNone = self.assertIsNotNone
            TestEmbedderPipeline.assertIsInstance = self.assertIsInstance
            TestEmbedderPipeline().main() # Call the original main as an instance method
            
    # Create a global class for the main logic if we want to use self.assert
class TestEmbedderPipeline: # Not inheriting from unittest.TestCase to avoid auto-discovery issues if not intended
    # Define assertion methods that will be bound if run via the DummyTest hack
    def assertEqual(self, first, second, msg=None): assert first == second, msg or f"{first} != {second}"
    def assertIsNone(self, obj, msg=None): assert obj is None, msg or f"{obj} is not None"
    def assertIsNotNone(self, obj, msg=None): assert obj is not None, msg or f"{obj} is None"
    def assertIsInstance(self, obj, cls, msg=None): assert isinstance(obj, cls), msg or f"{obj} is not instance of {cls}"
    
    # Keep your original main function here
    def main(self):
        status_msgs = []
        completion_info = {}

        def status_cb(message: str, current_step: Optional[int]=None, total_steps: Optional[int]=None, detail: Optional[str]=None):
            log_msg = f"STATUS CB: {message}"
            if current_step is not None and total_steps is not None: log_msg += f" ({current_step}/{total_steps})"
            if detail: log_msg += f" - {detail}"
            logging.info(log_msg)
            status_msgs.append(message)

        def final_cb(changed: bool):
            logging.info(f"FINAL CB: graph changed={changed}")
            completion_info['graph_change'] = changed

        if not create_test_config(): # This is your existing helper
            logging.critical("TEST SETUP Err EXIT NOW ")
            sys.exit(1)

        config_obj_read = None
        try:
            with open(TEST_CONFIG_FILE, "r") as cfg_r: # TEST_CONFIG_FILE is your existing global
                config_obj_read = json.load(cfg_r)
            logging.debug(f"LOADING config For TEST: {config_obj_read}")
        except Exception as e: 
            logging.exception(f"CANNOT Load Temp Config NEEDED!! {e}")
            cleanup_test_files() 
            sys.exit(1)

        dummy_vector_for_pipeline = [0.5, 0.4, 0.3, 0.2, 0.1]
        
        with patch('lucid_memory.embedder.requests.post') as mock_embedder_requests_post:
            mock_embedder_response = MagicMock()
            mock_embedder_response.raise_for_status = MagicMock()
            mock_embedder_response.json.return_value = {"data": [{"embedding": dummy_vector_for_pipeline}]}
            mock_embedder_requests_post.return_value = mock_embedder_response

            digestor = None; embedder = None
            try:
                digestor = Digestor(config_obj_read) 
                embedder = Embedder(config_obj_read)
                logging.info(f"----- Digestor Init = {digestor is not None} (Will attempt real LLM calls if backend is up) ----")
                logging.info(f"----- Embedder Init = {embedder.is_available() if embedder else False} (HTTP calls will be MOCKED) ---")
            except Exception as e:
                logging.exception(f"INIT components FAILED STOP!! {e}"); cleanup_test_files(); sys.exit(1)

            graph = MemoryGraph()
            processed_ok = True
            temp_py_file_path = "" # Define to ensure it's in scope for finally
            try:
                if not os.path.exists(os.path.dirname(TEST_CONFIG_FILE)): os.makedirs(os.path.dirname(TEST_CONFIG_FILE))
                temp_py_file_path = os.path.join(os.path.dirname(TEST_CONFIG_FILE), "test_pipeline_dummy_script.py")
                with open(temp_py_file_path, "w") as f: f.write(PY_CONTENT) # PY_CONTENT is your existing global
                chunks = chunk_file(temp_py_file_path, PY_CONTENT)

                if not chunks: logging.error("**** CHUNKER BROKEN?? No chunks.."); cleanup_test_files(); sys.exit(1)

                logging.info(f"OK GOT {len(chunks)} Chunks... Creating PROCESSOR....")
                processor = ChunkProcessor(
                    digestor=digestor, embedder=embedder, memory_graph=graph,
                    status_callback=status_cb, completion_callback=final_cb,
                    memory_graph_path_override=TEST_MEMORY_FILE # TEST_MEMORY_FILE is your existing global
                )
                processor.process_chunks(chunks, original_filename="dummy_script_for_pipeline.py")

                final_graph = MemoryGraph()
                final_graph.load_from_json(TEST_MEMORY_FILE)
                graph_nodes = list(final_graph.nodes.values())
                node_count = len(graph_nodes)

                self.assertGreater(node_count, 0, "FAILURE: NO Nodes created AT ALL in Graph.") # Use self.
                logging.info(f"PASS=> Created {node_count} Graph Nodes total SUCCESS!")

                found_embeddings = 0
                embedder_was_available_for_test = embedder.is_available() if embedder else False

                for node in graph_nodes:
                    self.assertIsNotNone(node.summary, f"Node {node.id} summary is missing.")
                    embData = getattr(node, 'embedding', None)
                    if embData:
                        self.assertIsInstance(embData, list, f"FAIL Node {node.id} EMBEDDING NOT List")
                        self.assertEqual(embData, dummy_vector_for_pipeline, f"Node {node.id} embedding mismatch")
                        found_embeddings += 1
                
                logging.info(f"FINAL CHECK => Found {found_embeddings} Mock-Embedded nodes. Embedder active={embedder_was_available_for_test}")
                if embedder_was_available_for_test:
                    self.assertEqual(found_embeddings, node_count, "Not all nodes got mock embeddings when embedder was available.")
                else:
                    self.assertEqual(found_embeddings, 0, "Embeddings found even when embedder was not available.")
                logging.info("======= MOCKED EMBEDDING PIPELINE TEST PASSED =======")
            except AssertionError:
                processed_ok = False; logging.exception("Pipeline Test Assertion FAILED.")
            except Exception as e:
                processed_ok = False; logging.exception(f"MAIN TEST EXECUTION FAILED: {e}")
            finally:
                cleanup_test_files()
                if temp_py_file_path and os.path.exists(temp_py_file_path):
                    try: os.remove(temp_py_file_path)
                    except OSError: pass
                if not processed_ok:
                    logging.error("======= EMBEDDING PIPELINE TEST FAILED ======="); sys.exit(1)

if __name__ == "__main__":
    # This allows running: python tests/test_embedding_pipeline.py
    # It uses the TestEmbedderPipeline class with its own assert methods.
    TestEmbedderPipeline().main()