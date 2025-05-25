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