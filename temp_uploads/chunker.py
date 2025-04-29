import ast
import re
import os
from typing import List, Dict, Any, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data.yaml")

# --- Python Code Chunking (using AST) ---

class PythonCodeChunker(ast.NodeVisitor):
    def __init__(self, source_code: str, file_identifier: str = "unknown_file"):
        self.source_lines = source_code.splitlines(keepends=True)
        self.chunks: List[Dict[str, Any]] = []
        self.current_class_name: Optional[str] = None
        self.file_identifier = file_identifier # Store filename for top-level parent ID
        self.sequence_counter = 0 # Counter for chunk order

    def _get_node_content(self, node: ast.AST) -> str:
        if not hasattr(node,'lineno')or not hasattr(node,'end_lineno'):
            return""

        start_line=node.lineno-1
        end_line=node.end_lineno

        if hasattr(node,'decorator_list')and node.decorator_list:
            try:
                min_decorator_line=min(d.lineno for d in node.decorator_list if hasattr(d,'lineno'))
                start_line=min(start_line,min_decorator_line-1)
            except ValueError:
                pass

        start_line=max(0,start_line)
        end_line=min(end_line,len(self.source_lines))
        return"".join(self.source_lines[start_line:end_line])


    def _add_chunk(self, content: str, node_type: str, identifier: str, node: ast.AST, parent_id: Optional[str]):
        """Helper to create and add a chunk dictionary."""
        if content:
            self.chunks.append({
                "content": content,
                "metadata": {
                    "type": node_type,
                    "identifier": identifier,
                    "parent_identifier": parent_id, # Added parent
                    "sequence_index": self.sequence_counter, # Added sequence
                    "start_line": getattr(node, 'lineno', None), # Use getattr for safety
                    "end_line": getattr(node, 'end_lineno', None)
                }
            })
            self.sequence_counter += 1 # Increment sequence number

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Handles function definitions."""
        identifier = f"{self.current_class_name}.{node.name}" if self.current_class_name else node.name
        chunk_content = self._get_node_content(node)
        # Parent is class name if inside a class, otherwise the file identifier
        parent = self.current_class_name or self.file_identifier
        self._add_chunk(chunk_content, "python_function", identifier, node, parent)
        # self.generic_visit(node) # Skip nested functions for now

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handles async function definitions."""
        identifier = f"{self.current_class_name}.{node.name}" if self.current_class_name else node.name
        chunk_content = self._get_node_content(node)
        parent = self.current_class_name or self.file_identifier
        self._add_chunk(chunk_content, "python_function_async", identifier, node, parent)
        # self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Handles class definitions and visits methods within them."""
        class_identifier = node.name
        # Decide if we want a node for the class itself (Optional)
        # If so, its parent would be the file identifier
        # class_content = self._get_node_content(node) # Get full class content
        # self._add_chunk(class_content, "python_class", class_identifier, node, self.file_identifier)

        # Process methods *within* the class
        original_class_name = self.current_class_name
        self.current_class_name = class_identifier # Set parent context for methods
        # Explicitly visit function defs within the class body
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item)
        self.current_class_name = original_class_name # Restore context


    def chunk(self) -> List[Dict[str, Any]]:
        """Parses the code and returns the collected chunks."""
        try:
            tree = ast.parse("".join(self.source_lines), type_comments=True)
            # Visit only top-level functions and classes initially
            for node in tree.body:
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                     self.visit(node)
            return self.chunks
        except SyntaxError as e:
             logging.error(f"AST Parsing Error in {self.file_identifier}: {e}. Cannot chunk accurately.", exc_info=True)
             # Fallback: return whole file as one chunk with limited metadata
             return [{
                 "content": "".join(self.source_lines),
                 "metadata": {"type": "python_file_parse_error", "identifier": self.file_identifier, "sequence_index": 0, "parent_identifier": None}
             }]


# --- Markdown Chunking ---

def chunk_markdown(text: str, file_identifier: str = "unknown_markdown") -> List[Dict[str, Any]]:
    """Chunks Markdown text by level 2 headers (##)."""
    chunks = []
    parts = re.split(r'(\n##\s+.*)', text) # Split before headers
    current_content = ""
    current_header = "Introduction" # Default header
    sequence_counter = 0 # Sequence index for markdown sections

    if parts and parts[0].strip():
        current_content = parts[0].strip()

    for i in range(1, len(parts), 2):
        header_line = parts[i].strip()
        content_after_header = parts[i+1] if (i + 1) < len(parts) else ""
        header_identifier = re.sub(r'^##\s*', '', header_line)

        if current_content: # Save previous section
            chunks.append({
                "content": current_content,
                "metadata": {
                    "type": "markdown_section",
                    "identifier": current_header,
                    "parent_identifier": file_identifier, # File is parent of sections
                    "sequence_index": sequence_counter
                }
            })
            sequence_counter += 1

        current_header = header_identifier
        current_content = content_after_header.strip() # Content for *this* header

    if current_content: # Add the last section
        chunks.append({
            "content": current_content,
            "metadata": {
                "type": "markdown_section",
                "identifier": current_header,
                "parent_identifier": file_identifier,
                "sequence_index": sequence_counter
            }
        })
        sequence_counter += 1

    # Handle case where there are no '##' headers
    if not chunks and text.strip():
         chunks.append({
             "content": text.strip(),
             "metadata": {"type": "markdown_file", "identifier": file_identifier, "parent_identifier": None, "sequence_index": 0}
         })

    logging.info(f"Chunked Markdown '{file_identifier}' into {len(chunks)} sections.")
    return chunks


# --- Plain Text Chunking ---

def chunk_plain_text(text: str, file_identifier: str = "unknown_text", max_chars: int = 1500) -> List[Dict[str, Any]]:
    """Chunks plain text by paragraphs, splitting large paragraphs."""
    chunks = []
    paragraphs = re.split(r'\n\s*\n', text)
    sequence_counter = 0

    for i, paragraph in enumerate(paragraphs):
        paragraph = paragraph.strip()
        if not paragraph: continue

        if len(paragraph) <= max_chars:
            chunks.append({
                "content": paragraph,
                "metadata": {"type": "text_paragraph", "identifier": f"Paragraph {i+1}", "parent_identifier": file_identifier, "sequence_index": sequence_counter}
            })
            sequence_counter += 1
        else:
            start, para_chunk_counter = 0, 1
            while start < len(paragraph):
                end = start + max_chars 
                split_pos = paragraph.rfind(' ', start, end)
                if split_pos != -1 and end < len(paragraph): end = split_pos + 1
                elif end >= len(paragraph): end = len(paragraph)
                chunk_content = paragraph[start:end].strip()
                if chunk_content:
                    chunks.append({"content": chunk_content, "metadata": {"type": "text_split", "identifier": f"Paragraph {i+1} Part {para_chunk_counter}", "parent_identifier": file_identifier, "sequence_index": sequence_counter}})
                    sequence_counter += 1
                    para_chunk_counter += 1
                start = end

    logging.info(f"Chunked Text '{file_identifier}' into {len(chunks)} paragraphs/splits.")
    return chunks


# --- Main Chunker Function ---

def chunk_file(file_path: str, file_content: str) -> List[Dict[str, Any]]:
    """Detects file type and applies the appropriate chunking strategy."""
    filename = os.path.basename(file_path)
    base_filename_noext, extension = os.path.splitext(filename)
    extension = extension.lower()

    logging.info(f"Chunking file: {filename} (type: {extension})")

    # Pass filename identifier to chunkers
    if extension == ".py":
        # Use base filename as the top-level parent identifier for functions
        chunker = PythonCodeChunker(file_content, file_identifier=base_filename_noext)
        return chunker.chunk()
    elif extension == ".md":
        return chunk_markdown(file_content, file_identifier=base_filename_noext)
    elif extension in [".txt", ".log"] or not extension:
         return chunk_plain_text(file_content, file_identifier=base_filename_noext)
    else:
        logging.warning(f"Unknown file type '{extension}'. Using plain text chunking.")
        return chunk_plain_text(file_content, file_identifier=base_filename_noext)


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    try:
        with open(TEST_DATA_PATH,"r")as f:test_data=yaml.safe_load(f)
        test_py_content=test_data.get('python_code','')
        test_md_content=test_data.get('markdown_text','')
        test_txt_content=test_data.get('plain_text','')
        logging.info(f"Loaded test data from {TEST_DATA_PATH}")
    except Exception as e: 
        logging.error(f"Error loading test data: {e}")

    test_py_content,test_md_content,test_txt_content="","",""

    if test_py_content:
        print("\n--- Testing Python Chunker ---")
        py_chunks = chunk_file("test.py", test_py_content)
        print(f"Found {len(py_chunks)} Python chunks.")
        for i, chunk in enumerate(py_chunks):
            meta = chunk.get('metadata', {})
            print(f"  Chunk {i+1}: Seq={meta.get('sequence_index')}, Type={meta.get('type')}, "
                  f"ID={meta.get('identifier')}, Parent={meta.get('parent_identifier')}, "
                  f"Lines {meta.get('start_line')}-{meta.get('end_line')}")
        print("-" * 30)

    if test_md_content:
        print("\n--- Testing Markdown Chunker ---")
        md_chunks = chunk_file("test.md", test_md_content)
        print(f"Found {len(md_chunks)} Markdown chunks.")
        for i, chunk in enumerate(md_chunks):
            meta = chunk.get('metadata', {})
            print(f"  Chunk {i+1}: Seq={meta.get('sequence_index')}, Type={meta.get('type')}, "
                  f"ID='{meta.get('identifier')}', Parent={meta.get('parent_identifier')}")
        print("-" * 30)

    if test_txt_content:
        print("\n--- Testing Plain Text Chunker ---")
        txt_chunks = chunk_file("test.txt", test_txt_content)
        print(f"Found {len(txt_chunks)} Text chunks.")
        for i, chunk in enumerate(txt_chunks):
             meta = chunk.get('metadata', {})
             print(f"  Chunk {i+1}: Seq={meta.get('sequence_index')}, Type={meta.get('type')}, "
                   f"ID='{meta.get('identifier')}', Parent={meta.get('parent_identifier')}")
        print("-" * 30)