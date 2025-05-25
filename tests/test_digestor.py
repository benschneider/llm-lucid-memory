import unittest
from unittest.mock import patch, MagicMock, mock_open
import requests
import yaml
import json

from lucid_memory.digestor import Digestor, DEFAULT_CONFIG_PATH_DIGESTOR, DEFAULT_PROMPTS_FILE_PATH_DIGESTOR
from lucid_memory.memory_node import MemoryNode

# Default dummy config for the Digestor
DUMMY_DIGESTOR_CONFIG = {
    "backend_url": "http://localhost:11434/v1/chat/completions",
    "model_name": "test-digestor-model",
    "api_key": "test_api_key_for_digestor"
}

# Dummy prompts content
DUMMY_PROMPTS_CONTENT = {
    'summary': "Summarize this: {raw_text}",
    'key_concepts': "Key concepts for: {raw_text}",
    'tags': "Tags for: {raw_text}",
    'questions': "Questions about: {raw_text}",
    'code_dependencies': "Dependencies for code: {code_chunk}",
    'code_outputs': "Outputs for code: {code_chunk} given dependencies: {dependency_list}",
    'code_key_variables': "Key variables/params for code: {code_chunk}"
}
DUMMY_PROMPTS_YAML = yaml.dump(DUMMY_PROMPTS_CONTENT)


class TestDigestor(unittest.TestCase):

    def setUp(self):
        # Patch 'open' for prompts.yaml and 'os.path.exists' for config file check
        # This ensures Digestor can initialize even if files are not physically present during tests
        # or if we want to control their content.
        self.mock_file_content = {
            DEFAULT_PROMPTS_FILE_PATH_DIGESTOR: DUMMY_PROMPTS_YAML,
            # We can also mock the config if Digestor tries to load its own
            DEFAULT_CONFIG_PATH_DIGESTOR: json.dumps(DUMMY_DIGESTOR_CONFIG)
        }

        def mock_open_side_effect(filename, *args, **kwargs):
            if filename in self.mock_file_content:
                return mock_open(read_data=self.mock_file_content[filename])()
            raise FileNotFoundError(f"File {filename} not found in mock.")

        self.open_patcher = patch('builtins.open', mock_open_side_effect)
        self.exists_patcher = patch('os.path.exists', return_value=True) # Assume files exist

        self.mock_open = self.open_patcher.start()
        self.mock_exists = self.exists_patcher.start()

    def tearDown(self):
        self.open_patcher.stop()
        self.exists_patcher.stop()

    def test_digestor_initialization_with_config_dict(self):
        """Test Digestor initializes correctly when a config dict is passed."""
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG)
        self.assertEqual(digestor.llm_url, DUMMY_DIGESTOR_CONFIG["backend_url"])
        self.assertEqual(digestor.model_name, DUMMY_DIGESTOR_CONFIG["model_name"])
        self.assertEqual(digestor.api_key, DUMMY_DIGESTOR_CONFIG["api_key"])
        self.assertEqual(digestor.prompts, DUMMY_PROMPTS_CONTENT)

    def test_digestor_initialization_loads_default_config(self):
        """Test Digestor loads its default config if none is passed (mocked file)."""
        # Ensure os.path.exists for DEFAULT_CONFIG_PATH_DIGESTOR returns true for this test path
        self.mock_exists.side_effect = lambda path: path == DEFAULT_CONFIG_PATH_DIGESTOR or path == DEFAULT_PROMPTS_FILE_PATH_DIGESTOR

        digestor = Digestor(config=None) # Trigger default loading
        self.assertEqual(digestor.llm_url, DUMMY_DIGESTOR_CONFIG["backend_url"])
        self.assertEqual(digestor.model_name, DUMMY_DIGESTOR_CONFIG["model_name"])
        self.assertEqual(digestor.api_key, DUMMY_DIGESTOR_CONFIG["api_key"])

    def test_format_prompt(self):
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG)
        prompt = digestor._format_prompt('summary', raw_text="test content")
        self.assertEqual(prompt, "Summarize this: test content")
        
        prompt_missing_key = digestor._format_prompt('non_existent_key')
        self.assertIsNone(prompt_missing_key)

        prompt_bad_placeholder = digestor._format_prompt('summary', wrong_placeholder="foo")
        self.assertIsNone(prompt_bad_placeholder) # Expect None due to logged error and return None

    @patch('lucid_memory.digestor.requests.post')
    def test_call_llm_success(self, mock_post):
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG)
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": " LLM Response "}}]
        }
        mock_post.return_value = mock_response

        response_content = digestor._call_llm("Test prompt", "Test Task")
        self.assertEqual(response_content, "LLM Response") # Stripped
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        self.assertEqual(args[0], DUMMY_DIGESTOR_CONFIG["backend_url"])
        self.assertEqual(kwargs['json']['model'], DUMMY_DIGESTOR_CONFIG["model_name"])
        self.assertIn('Authorization', kwargs['headers'])
        self.assertEqual(kwargs['headers']['Authorization'], f"Bearer {DUMMY_DIGESTOR_CONFIG['api_key']}")


    @patch('lucid_memory.digestor.requests.post')
    def test_call_llm_http_error(self, mock_post):
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG)
        mock_response = MagicMock(status_code=500, text="Server Error")
        mock_post.side_effect = requests.exceptions.HTTPError(response=mock_response)
        
        response_content = digestor._call_llm("Test prompt", "Test Task")
        self.assertIsNone(response_content)

    def test_parse_list_output(self):
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG) # Init for access to method
        
        # Test newline separated
        raw1 = "Item 1\nItem 2\n- Item 3\n* Item4"
        parsed1 = digestor._parse_list_output(raw1, "task1")
        self.assertEqual(parsed1, ["Item 1", "Item 2", "Item 3", "Item4"])

        # Test comma separated
        raw2 = "Apple, Banana, Cherry"
        parsed2 = digestor._parse_list_output(raw2, "task2")
        self.assertEqual(parsed2, ["Apple", "Banana", "Cherry"])

        # Test mixed, with quotes and numbers
        raw3 = '1. "First item"\n2. Second, also quoted\nNone' # "None" should be filtered
        parsed3 = digestor._parse_list_output(raw3, "task3")
        self.assertEqual(parsed3, ["First item", "Second, also quoted"])
        
        # Test empty or "None"
        self.assertEqual(digestor._parse_list_output(None, "task_none"), [])
        self.assertEqual(digestor._parse_list_output("  None  ", "task_none_str"), [])
        self.assertEqual(digestor._parse_list_output("N/A", "task_na"), [])
        self.assertEqual(digestor._parse_list_output("", "task_empty"), [])

        # Test single item that's not a list format
        raw4 = "Just one thing"
        parsed4 = digestor._parse_list_output(raw4, "task4")
        self.assertEqual(parsed4, ["Just one thing"])


    @patch.object(Digestor, '_call_llm') # Patching the method on the class
    def test_digest_text_chunk(self, mock_call_llm):
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG)
        
        # Define what _call_llm will return for each type of task
        def llm_side_effect(prompt, task_description, temperature=0.1, max_retries=1):
            if "Summarize" in prompt: return "Mocked Summary"
            if "Key concepts" in prompt: return "Concept A\nConcept B"
            if "Tags for" in prompt: return "tag1, tag2"
            if "Questions about" in prompt: return "Is it good?\nWhy is it so?"
            return None # Default for unhandled prompts
        
        mock_call_llm.side_effect = llm_side_effect

        raw_text = "This is a piece of text to digest."
        node_id = "text_node_01"
        chunk_meta = {"type": "text_paragraph"}

        memory_node = digestor.digest(raw_text, node_id, chunk_metadata=chunk_meta, generate_questions=True)

        self.assertIsNotNone(memory_node)
        self.assertIsInstance(memory_node, MemoryNode)
        self.assertEqual(memory_node.id, node_id)
        self.assertEqual(memory_node.raw, raw_text)
        self.assertEqual(memory_node.summary, "Mocked Summary")
        self.assertEqual(memory_node.key_concepts, ["Concept A", "Concept B"])
        self.assertEqual(memory_node.tags, ["tag1", "tag2"])
        self.assertEqual(memory_node.follow_up_questions, ["Is it good?", "Why is it so?"])
        self.assertEqual(memory_node.dependencies, []) # No code-specific calls for text
        self.assertEqual(memory_node.produced_outputs, [])

        # Check how many times _call_llm was invoked
        # Summary, Key Concepts, Tags, Questions = 4 calls
        self.assertEqual(mock_call_llm.call_count, 4)


    @patch.object(Digestor, '_call_llm')
    def test_digest_code_chunk(self, mock_call_llm):
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG)

        def llm_side_effect(prompt, task_description, temperature=0.1, max_retries=1):
            # logger.debug(f"Mock _call_llm for task: {task_description}, prompt starts with: {prompt[:50]}")
            if "Summarize" in prompt: return "Code Summary: Does stuff."
            if "Key variables/params for code" in prompt: return "param_x\nlocal_var_y"
            if "Tags for" in prompt: return "code, python, logic"
            if "Dependencies for code" in prompt: return "module_os\nexternal_api_call"
            if "Outputs for code" in prompt: return "returns_dataframe\nmodifies_global_z"
            return f"UNHANDLED MOCK PROMPT for {task_description}"

        mock_call_llm.side_effect = llm_side_effect

        raw_code = "def my_function(x):\n  import os\n  return os.getcwd()"
        node_id = "code_node_01"
        chunk_meta = {"type": "python_function"}

        memory_node = digestor.digest(raw_code, node_id, chunk_metadata=chunk_meta, generate_questions=False)

        self.assertIsNotNone(memory_node)
        self.assertEqual(memory_node.summary, "Code Summary: Does stuff.")
        self.assertEqual(memory_node.key_concepts, ["param_x", "local_var_y"]) # From 'code_key_variables' prompt
        self.assertEqual(memory_node.tags, ["code", "python", "logic"])
        self.assertEqual(memory_node.dependencies, ["module_os", "external_api_call"])
        self.assertEqual(memory_node.produced_outputs, ["returns_dataframe", "modifies_global_z"])
        self.assertEqual(memory_node.follow_up_questions, []) # generate_questions=False

        # Summary, Key Vars, Tags, Dependencies, Outputs = 5 calls
        self.assertEqual(mock_call_llm.call_count, 5)

    def test_digest_empty_raw_text(self):
        """Test that digest returns None for empty raw_text."""
        digestor = Digestor(config=DUMMY_DIGESTOR_CONFIG)
        memory_node = digestor.digest("", "empty_node_id")
        self.assertIsNone(memory_node)


if __name__ == '__main__':
    unittest.main()