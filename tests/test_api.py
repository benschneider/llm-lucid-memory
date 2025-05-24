import unittest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app instance from server_api.py
from lucid_memory.server_api import app as fastapi_app, lucid_controller as global_lucid_controller

class TestUnifiedAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(fastapi_app)
        # We need to mock the global lucid_controller instance used by server_api.py
        # This mock will be active for all tests in this class
        self.controller_mock = MagicMock()
        
        # Patch the global lucid_controller instance in the server_api module
        # This is tricky because server_api.py instantiates it directly.
        # A better approach for testing might be dependency injection for the controller.
        # For now, we'll try to patch where it's used or ensure tests don't rely on deep controller state
        # or we patch its methods directly.
        
        # Let's patch specific methods of the *actual* global_lucid_controller for simplicity if it's already loaded
        # Or, if server_api.lucid_controller can be replaced:
        self.patcher = patch('lucid_memory.server_api.lucid_controller', self.controller_mock)
        self.mocked_lucid_controller = self.patcher.start()


    def tearDown(self):
        self.patcher.stop()

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Welcome to Lucid Memory Unified Server API", response.json()["message"])

    def test_library_chunk_route(self):
        # Mock the library's chunk function directly as it's called by the route
        with patch('lucid_memory.server_api.lib_chunk_content') as mock_lib_chunk:
            mock_lib_chunk.return_value = [{'chunk1': 'content1'}, {'chunk2': 'content2'}]
            response = self.client.post('/library/chunk', json={'source_code': 'def hello(): print("Hello")'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {'chunks': [{'chunk1': 'content1'}, {'chunk2': 'content2'}]})
            mock_lib_chunk.assert_called_once_with('def hello(): print("Hello")', file_identifier="api_chunk_input.py")

    def test_library_process_route(self):
        mock_node = MagicMock(spec=MemoryNode) # Use spec for better mocking
        mock_node.to_dict.return_value = {'id': 'node1', 'content': 'processed_content'}
        self.mocked_lucid_controller.process_chunks_synchronously.return_value = [mock_node]
        
        payload = {'chunks': [{'content': 'abc'}], 'original_filename': 'test.py'}
        response = self.client.post('/library/process', json=payload)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'memory_nodes': [{'id': 'node1', 'content': 'processed_content'}]})
        self.mocked_lucid_controller.process_chunks_synchronously.assert_called_once_with(
            payload['chunks'], payload['original_filename']
        )

    def test_library_add_memory_node_route(self):
        self.mocked_lucid_controller.add_memory_node_data.return_value = True
        payload = {'memory_node': {'id': 'node1', 'raw': 'test raw'}}
        response = self.client.post('/library/nodes/add', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'success': True, 'message': "Memory node added successfully."})
        self.mocked_lucid_controller.add_memory_node_data.assert_called_once_with(payload['memory_node'])

    def test_library_get_memory_nodes_route(self):
        mock_node = MagicMock(spec=MemoryNode)
        mock_node.to_dict.return_value = {'id': 'node1', 'content': 'node_content'}
        self.mocked_lucid_controller.get_all_memory_nodes.return_value = {'node1': mock_node}
        response = self.client.get('/library/nodes')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'memory_nodes': [{'id': 'node1', 'content': 'node_content'}]})

    def test_library_update_config_route(self):
        self.mocked_lucid_controller.update_config_values.return_value = True
        self.mocked_lucid_controller.config_mgr.get_config.return_value = {"version": "0.3.0"} # Mock for server_config update
        payload = {'new_config_values': {'key': 'value'}}
        response = self.client.post('/library/config/update', json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {'success': True, 'message': "Configuration updated successfully. Components reloaded."})
        self.mocked_lucid_controller.update_config_values.assert_called_once_with(payload['new_config_values'])

    def test_library_status_route(self):
        # Setup mocks for the controller methods called by the status route
        self.mocked_lucid_controller.is_processing_active.return_value = False
        self.mocked_lucid_controller.get_last_status.return_value = "System Ready"
        self.mocked_lucid_controller.get_config.return_value = {
            'api_type': 'OllamaTest', 
            'model_name': 'digest_model', 
            'backend_url': 'http://digest',
            'embedding_api_model_name': 'embed_model'
        }
        self.mocked_lucid_controller.is_digestor_ready = True # Property mock
        self.mocked_lucid_controller.is_embedder_ready = True # Property mock
        
        # Mock the graph nodes access
        mock_graph = MagicMock()
        mock_graph.nodes = {'node1': None, 'node2': None} # Simulate 2 nodes
        self.mocked_lucid_controller.memory_graph = mock_graph

        # Patch server_config temporarily for this test if it's complex to mock its global update
        with patch('lucid_memory.server_api.server_config', {
            "backend_url": "http://proxy_backend",
            "model_name": "proxy_model",
            "embedding_api_model_name": "proxy_embed_model",
            "api_key": "fakekey"
        }):
            response = self.client.get('/library/status')
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['status'], "Lucid Memory Library API is running.")
        self.assertFalse(data['processing_active'])
        self.assertEqual(data['last_controller_status'], "System Ready")
        self.assertEqual(data['controller_config']['model_name_for_digestor'], 'digest_model')
        self.assertTrue(data['controller_config']['digestor_ready'])
        self.assertEqual(data['proxy_config']['model_name'], 'proxy_model')
        self.assertTrue(data['proxy_config']['api_key_present_for_proxy'])
        self.assertEqual(data['memory_graph_nodes'], 2)


    @patch('lucid_memory.server_api.requests.post') # Mock the actual HTTP call
    @patch('lucid_memory.server_api.retriever_for_proxy') # Mock the retriever
    def test_llm_proxy_chat_completions_route(self, mock_retriever, mock_requests_post):
        # Setup mock for retriever
        mock_node = MagicMock(spec=MemoryNode)
        mock_node.id = "retrieved_node1"
        mock_node.sequence_index = 0
        mock_node.parent_identifier = "file.py"
        mock_node.summary = "This is a test node."
        mock_node.key_concepts = ["testing", "mocking"]
        mock_node.dependencies = []
        mock_node.produced_outputs = []
        mock_node.tags = ["test"]
        mock_retriever.retrieve_by_keyword.return_value = [mock_node]
        mock_retriever.reflect_on_candidates.return_value = [mock_node]

        # Setup mock for requests.post (backend LLM call)
        mock_backend_response = MagicMock()
        mock_backend_response.status_code = 200
        mock_backend_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "This is the LLM response."}
            }]
        }
        mock_requests_post.return_value = mock_backend_response

        payload = {
            "model": "client_requested_model",
            "messages": [{"role": "user", "content": "Hello, world?"}]
        }
        response = self.client.post('/v1/chat/completions', json=payload)

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data['choices'][0]['message']['content'], "This is the LLM response.")
        # Check that retriever was called
        mock_retriever.retrieve_by_keyword.assert_called_once_with("Hello, world?")
        # Check that requests.post was called (implies prompt augmentation happened)
        mock_requests_post.assert_called_once()
        # You could add more detailed assertions on the payload sent to the backend LLM if needed

if __name__ == '__main__':
    unittest.main()