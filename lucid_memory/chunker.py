import ast
import re
import os
from typing import List, Dict, Any, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.yaml") # Path for test data

# --- Python Code Chunking (using AST) ---

class PythonCodeChunker(ast.NodeVisitor):
    """
    Uses AST to chunk Python code into functions and classes.
    """
    def __init__(self, source_code: str):
        self.source_lines = source_code.splitlines(keepends=True)
        self.chunks: List[Dict[str, Any]] = []
        self.current_class_name: Optional[str] = None

    def _get_node_content(self, node: ast.AST) -> str:
        """Extracts the source code corresponding to an AST node, including decorators."""
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
            return ""

        start_line = node.lineno - 1 # Default start (0-based)
        end_line = node.end_lineno # AST gives 1-based end line, slice is exclusive

        # Adjust start line upwards to include decorators if present
        if hasattr(node, 'decorator_list') and node.decorator_list:
            # Decorators often start on lines before the function/class definition line
            # We need the earliest line number from the decorators
            try: # Handle potential errors if decorator node lacks lineno somehow
                min_decorator_line = min(d.lineno for d in node.decorator_list if hasattr(d, 'lineno'))
                start_line = min(start_line, min_decorator_line - 1)
            except ValueError: # No decorators had line numbers? Fallback.
                pass

        # Ensure start_line is not negative
        start_line = max(0, start_line)

        # Ensure end_line is within bounds
        end_line = min(end_line, len(self.source_lines))

        return "".join(self.source_lines[start_line:end_line])

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handles function definitions."""
        identifier = f"{self.current_class_name}.{node.name}" if self.current_class_name else node.name
        chunk_content = self._get_node_content(node)
        if chunk_content:
            self.chunks.append({
                "content": chunk_content,
                "metadata": {
                    "type": "python_function",
                    "identifier": identifier,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno
                }
            })
        # self.generic_visit(node) # Avoid double-counting nested functions

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handles async function definitions."""
        identifier = f"{self.current_class_name}.{node.name}" if self.current_class_name else node.name
        chunk_content = self._get_node_content(node)
        if chunk_content:
            self.chunks.append({
                "content": chunk_content,
                "metadata": {
                    "type": "python_function_async",
                    "identifier": identifier,
                    "start_line": node.lineno,
                    "end_line": node.end_lineno
                }
            })
        # self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Handles class definitions and visits methods within them."""
        class_content = self._get_node_content(node)
        class_identifier = node.name

        # Option 1: Add the class docstring/header as a separate chunk (if desired)
        # Find the end of the class header (usually the line with ':')
        # class_header_end_line = node.lineno
        # class_header_content = "".join(self.source_lines[node.lineno-1:class_header_end_line])
        # self.chunks.append({
        #     "content": class_header_content, # Or extract just docstring using ast.get_docstring(node)
        #     "metadata": {"type": "python_class_header", "identifier": class_identifier, ...}
        # })

        # Option 2: Just visit methods inside (current)
        original_class_name = self.current_class_name
        self.current_class_name = class_identifier
        # Iterate through the body of the class to find methods/functions
        for body_item in node.body:
            if isinstance(body_item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(body_item) # Visit methods directly
            # Could add handling for nested classes here if needed
        self.current_class_name = original_class_name

    def chunk(self) -> List[Dict[str, Any]]:
        """Parses the code and returns the collected chunks."""
        try:
            # Use type_comments=True for slightly better end_lineno in some Python versions
            tree = ast.parse("".join(self.source_lines), type_comments=True)
            self.visit(tree)
            # Add top-level code as a chunk? Might be useful.
            # Find code not inside any function/class
            # top_level_code = extract_top_level_code(tree, self.source_lines)
            # if top_level_code:
            #     self.chunks.insert(0, {"content": top_level_code, "metadata": {"type": "python_module_code"}})
            return self.chunks
        except SyntaxError as e:
             logging.error(f"AST Parsing Error: {e}. Cannot chunk Python file accurately.", exc_info=True)
             return [{ "content": "".join(self.source_lines), "metadata": {"type": "python_file_parse_error", "identifier": "unknown"} }]


# --- Markdown Chunking ---

def chunk_markdown(text: str) -> List[Dict[str, Any]]:
    """Chunks Markdown text by level 2 headers (##)."""
    chunks = []
    # Improved regex to handle optional space after ## and capture header text cleanly
    # It looks for \n## followed by space(s) or not, captures the header text until newline
    parts = re.split(r'(\n##\s+.*)', text) # Split *before* the header line

    current_content = ""
    current_header = "Introduction" # Default for content before the first header

    # First part is the intro content (if any)
    if parts and parts[0].strip():
        current_content = parts[0].strip()

    # Process pairs of (header_line, content_after_header)
    for i in range(1, len(parts), 2):
        # Header line includes the \n## prefix, content is everything after
        header_line = parts[i].strip() # e.g., "## Section One Header"
        content_after_header = parts[i+1] if (i + 1) < len(parts) else ""

        # Extract clean header identifier
        header_identifier = re.sub(r'^##\s*', '', header_line)

        # Save the *previous* chunk (content associated with the previous header)
        if current_content: # Avoid adding empty intro chunk if intro was empty
            chunks.append({
                "content": current_content, # Already stripped
                "metadata": {"type": "markdown_section", "identifier": current_header}
            })

        # Start the new chunk with the content *after* the current header
        current_header = header_identifier
        current_content = content_after_header.strip() # Content for *this* header

    # Add the very last chunk
    if current_content:
        chunks.append({
            "content": current_content,
            "metadata": {"type": "markdown_section", "identifier": current_header}
        })

    # Handle case where there are no '##' headers at all
    if not chunks and text.strip():
         chunks.append({
             "content": text.strip(),
             "metadata": {"type": "markdown_file", "identifier": "Full Document"}
         })

    logging.info(f"Chunked Markdown into {len(chunks)} sections.")
    return chunks


# --- Plain Text Chunking ---

def chunk_plain_text(text: str, max_chars: int = 1500) -> List[Dict[str, Any]]:
    """Chunks plain text by paragraphs, splitting large paragraphs."""
    # (This function remains the same)
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph: continue
        if len(paragraph) <= max_chars:
            chunks.append({"content": paragraph, "metadata": {"type": "text_paragraph", "identifier": f"Paragraph {i+1}"}})
        else:
            start, para_chunk_counter = 0, 1
            while start < len(paragraph):
                end = start + max_chars
                split_pos = paragraph.rfind(' ', start, end)
                if split_pos != -1 and end < len(paragraph): end = split_pos + 1
                elif end >= len(paragraph): end = len(paragraph)
                chunk_content = paragraph[start:end].strip()
                if chunk_content:
                    chunks.append({"content": chunk_content, "metadata": {"type": "text_split", "identifier": f"Paragraph {i+1} Part {para_chunk_counter}"}})
                    para_chunk_counter += 1
                start = end
    logging.info(f"Chunked Text into {len(chunks)} paragraphs/splits.")
    return chunks


# --- Main Chunker Function ---

def chunk_file(file_path: str, file_content: str) -> List[Dict[str, Any]]:
    """
    Detects file type and applies the appropriate chunking strategy.
    """
    # (This function remains the same)
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()
    logging.info(f"Chunking file: {os.path.basename(file_path)} (type: {extension})")
    if extension == ".py":
        chunker = PythonCodeChunker(file_content); return chunker.chunk()
    elif extension == ".md":
        return chunk_markdown(file_content)
    elif extension in [".txt", ".log"] or not extension:
         return chunk_plain_text(file_content)
    else:
        logging.warning(f"Unknown file type '{extension}'. Using plain text chunking.")
        return chunk_plain_text(file_content)


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    # Load test data from YAML
    try:
        with open(TEST_DATA_PATH, "r", encoding="utf-8") as f:
            test_data = yaml.safe_load(f)
        test_py_content = test_data.get('python_code', '')
        test_md_content = test_data.get('markdown_text', '')
        test_txt_content = test_data.get('plain_text', '')
        logging.info(f"Loaded test data from {TEST_DATA_PATH}")
    except FileNotFoundError:
        logging.error(f"Test data file not found at {TEST_DATA_PATH}. Cannot run tests.")
        test_py_content, test_md_content, test_txt_content = "", "", "" # Assign empty defaults
    except yaml.YAMLError as e:
         logging.error(f"Error parsing test data YAML {TEST_DATA_PATH}: {e}. Cannot run tests.", exc_info=True)
         test_py_content, test_md_content, test_txt_content = "", "", ""
    except Exception as e:
        logging.error(f"Error loading test data from {TEST_DATA_PATH}: {e}. Cannot run tests.", exc_info=True)
        test_py_content, test_md_content, test_txt_content = "", "", ""

    # Proceed with tests only if content was loaded
    if test_py_content:
        print("\n--- Testing Python Chunker ---")
        py_chunks = chunk_file("test.py", test_py_content)
        print(f"Found {len(py_chunks)} Python chunks.")
        for i, chunk in enumerate(py_chunks):
            print(f"  Chunk {i+1}: Type={chunk['metadata'].get('type')}, ID={chunk['metadata'].get('identifier')}, "
                  f"Lines {chunk['metadata'].get('start_line')}-{chunk['metadata'].get('end_line')}")
            # print(f"    Content:\n{chunk['content'][:100]}...") # Uncomment for content preview
        print("-" * 30)

    if test_md_content:
        print("\n--- Testing Markdown Chunker ---")
        md_chunks = chunk_file("test.md", test_md_content)
        print(f"Found {len(md_chunks)} Markdown chunks.")
        for i, chunk in enumerate(md_chunks):
            print(f"  Chunk {i+1}: ID={chunk['metadata'].get('identifier')}")
            # print(f"    Content:\n{chunk['content'][:100]}...")
        print("-" * 30)

    if test_txt_content:
        print("\n--- Testing Plain Text Chunker ---")
        txt_chunks = chunk_file("test.txt", test_txt_content)
        print(f"Found {len(txt_chunks)} Text chunks.")
        for i, chunk in enumerate(txt_chunks):
            print(f"  Chunk {i+1}: ID={chunk['metadata'].get('identifier')}")
        print("-" * 30)