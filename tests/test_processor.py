import unittest
from unittest.mock import patch, MagicMock, call, mock_open
import os
import time # For any sleep/timing related checks if necessary
import json # For checking saved graph content

# Components to be mocked or used
from lucid_memory.processor import ChunkProcessor, DEFAULT_MEMORY_GRAPH_PATH_PROC
from lucid_memory.digestor import Digestor
from lucid_memory.embedder import Embedder
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.memory_node import MemoryNode

# Dummy config that would be used by Controller to init Digestor/Embedder
DUMMY_COMPONENT_CONFIG = {
    "backend_url": "http://mock-llm-backend.com/v1",
    "model_name": "mock-model",
    "embedding_api_model_name": "mock-embedding-model",
    "api_key": "mock_key"
}

# Dummy prompts for Digestor mock if it needs prompts for some reason
DUMMY_PROMPTS_CONTENT_FOR_PROCESSOR_TEST = {
    'summary': "Summarize: {raw_text}", 'key_concepts': "Concepts: {raw_text}",
    'tags': "Tags: {raw_text}", 'questions': "Questions: {raw_text}",
    'code_dependencies': "Deps: {code_chunk}", 'code_outputs': "Outs: {code_chunk}",
    'code_key_variables': "Vars: {code_chunk}"
}

class TestChunkProcessor(unittest.TestCase):

    def setUp(self):
        # Mock Digestor and Embedder instances
        self.mock_digestor = MagicMock(spec=Digestor)
        self.mock_embedder = MagicMock(spec=Embedder)
        
        # Configure mock_embedder.is_available()
        self.mock_embedder.is_available.return_value = True # Assume available by default
        # Configure mock_embedder.generate_embedding to return a dummy vector or None
        self.dummy_embedding_vector = [0.1] * 10
        self.mock_embedder.generate_embedding.return_value = self.dummy_embedding_vector

        # Mock MemoryGraph instance
        self.mock_memory_graph = MagicMock(spec=MemoryGraph)
        self.mock_memory_graph.nodes = {} # Simulate the nodes dictionary
        
        # Mock callbacks
        self.mock_status_callback = MagicMock()
        self.mock_completion_callback = MagicMock()

        # Patch 'open' for the graph saving part and 'os.path.exists'
        self.open_patcher = patch('builtins.open', new_callable=mock_open)
        self.exists_patcher = patch('os.path.exists', return_value=False) # Default to file not existing
        
        self.mocked_open = self.open_patcher.start()
        self.mocked_os_exists = self.exists_patcher.start()

        # Ensure DEFAULT_MEMORY_GRAPH_PATH_PROC is defined for the test context
        # This is the default path processor will try to save to if no override.
        self.test_graph_save_path = DEFAULT_MEMORY_GRAPH_PATH_PROC


    def tearDown(self):
        self.open_patcher.stop()
        self.exists_patcher.stop()

    def _create_sample_chunks(self, count=2):
        chunks = []
        for i in range(count):
            chunks.append({
                "content": f"This is content for chunk {i+1}.",
                "metadata": {
                    "type": "text_paragraph",
                    "identifier": f"Para_{i+1}",
                    "parent_identifier": "test_document",
                    "sequence_index": i
                }
            })
        return chunks

    def _create_sample_node(self, node_id, raw_text, chunk_meta):
        # Helper to create a node similar to what Digestor would return
        return MemoryNode(
            id=node_id,
            raw=raw_text,
            summary=f"Summary for {node_id}",
            key_concepts=[f"concept_{node_id}"],
            tags=[f"tag_{node_id}"],
            sequence_index=chunk_meta.get("sequence_index"),
            parent_identifier=chunk_meta.get("parent_identifier")
            # embedding will be set by processor if embedder is available
        )

    def test_process_chunks_empty_list(self):
        """Test processing an empty list of chunks."""
        processor = ChunkProcessor(
            digestor=self.mock_digestor,
            embedder=self.mock_embedder,
            memory_graph=self.mock_memory_graph,
            status_callback=self.mock_status_callback,
            completion_callback=self.mock_completion_callback,
            memory_graph_path_override=self.test_graph_save_path
        )
        processor.process_chunks([], "empty_file.txt")

        self.mock_digestor.digest.assert_not_called()
        self.mock_embedder.generate_embedding.assert_not_called()
        self.mock_memory_graph.add_node.assert_not_called()
        self.mock_memory_graph.save_to_json.assert_not_called()
        
        # Check callbacks
        self.mock_status_callback.assert_called_with(
            message="Status: Finished 'empty_file.txt' - No chunks to process.",
            current_step=0, total_steps=0, detail=""
        )
        self.mock_completion_callback.assert_called_with(False) # Graph not changed

    def test_process_chunks_successful_flow_with_embedding(self):
        """Test the normal flow where chunks are digested and embedded."""
        sample_chunks = self._create_sample_chunks(2)
        
        # Configure mock_digestor.digest to return a MemoryNode
        # It will be called for each chunk.
        created_nodes = []
        def digest_side_effect(raw_text, node_id, chunk_metadata, generate_questions=False):
            # We use the passed node_id, or processor generates one.
            # Let's assume processor generates it based on its logic for this test.
            # The node_id passed to digest is what we need to check.
            # For simplicity in checking calls, let's make node_id predictable.
            meta_id = chunk_metadata.get("identifier", "unknown")
            gen_node_id = f"file_test_doc_text_paragraph_{meta_id}_{chunk_metadata.get('sequence_index', 0)+1}" # Approximate processor's ID
            
            node = self._create_sample_node(gen_node_id, raw_text, chunk_metadata)
            created_nodes.append(node)
            return node
        self.mock_digestor.digest.side_effect = digest_side_effect

        processor = ChunkProcessor(
            digestor=self.mock_digestor,
            embedder=self.mock_embedder, # Assumed available and returns dummy_embedding_vector
            memory_graph=self.mock_memory_graph,
            status_callback=self.mock_status_callback,
            completion_callback=self.mock_completion_callback,
            memory_graph_path_override=self.test_graph_save_path
        )
        processor.process_chunks(sample_chunks, "test_doc.txt")

        # Assertions
        self.assertEqual(self.mock_digestor.digest.call_count, len(sample_chunks))
        self.assertEqual(self.mock_embedder.generate_embedding.call_count, len(sample_chunks))
        self.assertEqual(self.mock_memory_graph.add_node.call_count, len(sample_chunks))
        
        # Check that the nodes added to the graph have embeddings
        for i, mock_call in enumerate(self.mock_memory_graph.add_node.call_args_list):
            added_node = mock_call.args[0]
            self.assertIsNotNone(added_node.embedding)
            self.assertEqual(added_node.embedding, self.dummy_embedding_vector)
            self.assertEqual(added_node.source, "test_doc.txt") # Check source filename
            self.assertEqual(added_node.sequence_index, sample_chunks[i]['metadata']['sequence_index'])
            self.assertEqual(added_node.parent_identifier, sample_chunks[i]['metadata']['parent_identifier'])


        self.mock_memory_graph.save_to_json.assert_called_once_with(self.test_graph_save_path)
        
        # Check final status and completion callbacks
        # The exact message might vary, focus on key parts or total steps
        last_status_call = self.mock_status_callback.call_args_list[-1]
        self.assertIn("Finished 'test_doc.txt'", last_status_call.kwargs['message'])
        self.assertEqual(last_status_call.kwargs['current_step'], len(sample_chunks))
        self.assertEqual(last_status_call.kwargs['total_steps'], len(sample_chunks))

        self.mock_completion_callback.assert_called_with(True) # Graph changed and saved

    def test_process_chunks_embedder_not_available(self):
        """Test flow when embedder is not available."""
        self.mock_embedder.is_available.return_value = False # Simulate embedder off
        sample_chunks = self._create_sample_chunks(1)
        
        node_id_from_proc = "file_test_doc2_text_paragraph_Para_1_1" # Example ID processor might make
        mock_node_from_digestor = self._create_sample_node(node_id_from_proc, sample_chunks[0]['content'], sample_chunks[0]['metadata'])
        self.mock_digestor.digest.return_value = mock_node_from_digestor

        processor = ChunkProcessor(
            digestor=self.mock_digestor,
            embedder=self.mock_embedder,
            memory_graph=self.mock_memory_graph,
            status_callback=self.mock_status_callback,
            completion_callback=self.mock_completion_callback,
            memory_graph_path_override=self.test_graph_save_path
        )
        processor.process_chunks(sample_chunks, "test_doc2.txt")

        self.mock_digestor.digest.assert_called_once()
        self.mock_embedder.generate_embedding.assert_not_called() # Key check
        self.mock_memory_graph.add_node.assert_called_once()
        
        added_node = self.mock_memory_graph.add_node.call_args.args[0]
        self.assertIsNone(added_node.embedding) # Embedding should be None

        self.mock_memory_graph.save_to_json.assert_called_once_with(self.test_graph_save_path)
        self.mock_completion_callback.assert_called_with(True)


    def test_process_chunks_digestor_returns_none(self):
        """Test flow when digestor fails for a chunk."""
        self.mock_digestor.digest.return_value = None # Simulate digestor failing
        sample_chunks = self._create_sample_chunks(1)

        processor = ChunkProcessor(
            digestor=self.mock_digestor,
            embedder=self.mock_embedder,
            memory_graph=self.mock_memory_graph,
            status_callback=self.mock_status_callback,
            completion_callback=self.mock_completion_callback,
            memory_graph_path_override=self.test_graph_save_path
        )
        processor.process_chunks(sample_chunks, "test_doc3.txt")

        self.mock_digestor.digest.assert_called_once()
        self.mock_embedder.generate_embedding.assert_not_called() # Not called if node is None
        self.mock_memory_graph.add_node.assert_not_called() # No node to add
        self.mock_memory_graph.save_to_json.assert_not_called() # Graph not changed
        self.mock_completion_callback.assert_called_with(False) # Graph not changed

        # Check status callback for failure indication in detail
        # This requires inspecting call_args_list for the detail message of the processed chunk
        processed_status_call = None
        for c in self.mock_status_callback.call_args_list:
            if c.kwargs.get('current_step') == 1 and c.kwargs.get('total_steps') == 1:
                processed_status_call = c
                break
        self.assertIsNotNone(processed_status_call, "Status for the processed chunk not found.")
        self.assertIn("FAILED/Skipped", processed_status_call.kwargs.get('detail', ""))


    def test_process_chunks_graph_save_failure(self):
        """Test flow when saving the graph fails."""
        sample_chunks = self._create_sample_chunks(1)
        node_id_from_proc = "file_test_doc4_text_paragraph_Para_1_1"
        mock_node_from_digestor = self._create_sample_node(node_id_from_proc, sample_chunks[0]['content'], sample_chunks[0]['metadata'])
        self.mock_digestor.digest.return_value = mock_node_from_digestor
        
        self.mock_memory_graph.save_to_json.side_effect = IOError("Disk full") # Simulate save error

        processor = ChunkProcessor(
            digestor=self.mock_digestor,
            embedder=self.mock_embedder,
            memory_graph=self.mock_memory_graph,
            status_callback=self.mock_status_callback,
            completion_callback=self.mock_completion_callback,
            memory_graph_path_override=self.test_graph_save_path
        )
        processor.process_chunks(sample_chunks, "test_doc4.txt")

        self.mock_memory_graph.add_node.assert_called_once()
        self.mock_memory_graph.save_to_json.assert_called_once_with(self.test_graph_save_path)
        
        # Completion callback should indicate graph change was attempted but save failed.
        # The current ChunkProcessor calls completion_callback(graph_changed_in_this_run and save_successful)
        # So, if save_successful is False, it will be completion_callback(False)
        self.mock_completion_callback.assert_called_with(False)

        # Check status callback for save error alert (optional, depends on processor's exact logging)
        save_error_alert_found = False
        for c_args, c_kwargs in self.mock_status_callback.call_args_list:
            if "ALERT: CRITICAL ERROR saving Memory Graph" in c_kwargs.get('message', ''):
                save_error_alert_found = True
                break
        # This specific alert is in _save_graph, let's check the final status message from process_chunks
        final_status_call = self.mock_status_callback.call_args_list[-1]
        self.assertIn("Graph saved: FAILED!", final_status_call.kwargs['message'])


if __name__ == '__main__':
    unittest.main()