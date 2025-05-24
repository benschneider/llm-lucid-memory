import ast
import re
import os
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)
# BasicConfig should be set at application entry point
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Expose functions for API
__all__ = ['chunk_file', 'chunk'] # 'chunk' is the function for raw string input

# --- Python Code Chunking (using AST) ---

class PythonCodeChunker(ast.NodeVisitor):
    def __init__(self, source_code: str, file_identifier: str = "unknown_file.py"): # Ensure .py for consistency
        self.source_lines = source_code.splitlines(keepends=True) # Keep newlines for accurate content
        self.chunks: List[Dict[str, Any]] = []
        self.current_class_name: Optional[str] = None
        # Ensure file_identifier is just the base name for parent_id context
        self.file_identifier_for_parent = os.path.splitext(file_identifier)[0]
        self.sequence_counter = 0

    def _get_node_content(self, node: ast.AST) -> str:
        # Ensure node has line numbers, otherwise it's a synthetic node or issue
        if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno') or \
           node.lineno is None or node.end_lineno is None:
            logger.warning(f"AST node '{getattr(node, 'name', type(node).__name__)}' missing line numbers in {self.file_identifier_for_parent}. Cannot get content.")
            return ""

        start_line = node.lineno - 1 
        end_line = node.end_lineno # end_lineno is inclusive of the last line of the node

        # Include decorators if present, by adjusting start_line
        if hasattr(node, 'decorator_list') and node.decorator_list:
            try:
                # Decorators can be complex; find the earliest line number among them
                min_decorator_line = min(d.lineno for d in node.decorator_list if hasattr(d, 'lineno') and d.lineno is not None)
                start_line = min(start_line, min_decorator_line - 1)
            except (ValueError, TypeError): # If no valid decorator lineno
                logger.debug(f"Could not determine decorator start line for node {getattr(node, 'name', type(node).__name__)}.")
                pass 
        
        start_line = max(0, start_line) # Ensure start_line is not negative
        # end_line is already the correct index for slicing (exclusive) if using source_lines list
        return "".join(self.source_lines[start_line:end_line]).strip()


    def _add_chunk(self, content: str, node_type: str, identifier: str, node: ast.AST, parent_id: Optional[str]):
        if content and content.strip(): # Only add if there's actual content
            # Clean up identifier: remove potential leading/trailing dots if class name was None
            clean_identifier = identifier.strip('.')
            
            # Get precise start/end, ensure they are valid
            start_line_num = getattr(node, 'lineno', None)
            end_line_num = getattr(node, 'end_lineno', None)

            # If decorator adjusted start line for content, metadata should reflect actual node start
            actual_node_start_line = getattr(node, 'lineno', None)


            self.chunks.append({
                "content": content,
                "metadata": {
                    "type": node_type, # e.g. "python_function", "python_class"
                    "identifier": clean_identifier, # e.g. "my_function", "MyClass.my_method"
                    "parent_identifier": parent_id, # e.g. "MyClass" or "my_module_name"
                    "sequence_index": self.sequence_counter,
                    "start_line": actual_node_start_line, 
                    "end_line": end_line_num
                }
            })
            self.sequence_counter += 1
        else:
            logger.debug(f"Skipped adding empty chunk for identifier '{identifier}' of type '{node_type}'.")


    def visit_FunctionDef(self, node: ast.FunctionDef):
        # For top-level functions, parent is file. For methods, parent is class.
        parent_context = self.current_class_name or self.file_identifier_for_parent
        func_identifier = f"{self.current_class_name}.{node.name}" if self.current_class_name else node.name
        
        chunk_content = self._get_node_content(node)
        self._add_chunk(chunk_content, "python_function", func_identifier, node, parent_context)
        # Do not call self.generic_visit(node) to avoid chunking nested functions/classes separately for now.
        # If nested items are desired as independent chunks, call generic_visit or manually visit them.

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        parent_context = self.current_class_name or self.file_identifier_for_parent
        func_identifier = f"{self.current_class_name}.{node.name}" if self.current_class_name else node.name
        
        chunk_content = self._get_node_content(node)
        self._add_chunk(chunk_content, "python_async_function", func_identifier, node, parent_context) # Differentiated type

    def visit_ClassDef(self, node: ast.ClassDef):
        class_identifier = node.name
        
        # Option 1: Create a chunk for the class definition itself (e.g., class MyClass: ...docstring... pass)
        # This content would typically be just the class structure without method bodies if methods are chunked separately.
        # For now, let's get the full class content for its own node.
        # class_shell_content = self._get_node_content(node) # This gets EVERYTHING including methods
        # A more refined "class shell" would parse out method bodies or use up to first method.
        # For simplicity, we can skip a dedicated "class shell" chunk if methods are the primary interest.
        # If we *do* want a class chunk:
        # self._add_chunk(class_shell_content, "python_class_definition", class_identifier, node, self.file_identifier_for_parent)

        # Option 2: Iterate through the body of the class to find methods
        original_class_context = self.current_class_name
        self.current_class_name = class_identifier # Set context for methods within this class
        
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.visit(item) # This will call visit_FunctionDef/visit_AsyncFunctionDef
            # Could also decide to visit nested ClassDefs if desired:
            # elif isinstance(item, ast.ClassDef):
            # self.visit(item) # For nested classes
        
        self.current_class_name = original_class_context # Restore context after visiting class body


    def chunk(self) -> List[Dict[str, Any]]:
        logger.info(f"PythonCodeChunker: Starting AST parsing for '{self.file_identifier_for_parent}'")
        try:
            # type_comments=True helps with some annotations but not strictly needed for basic chunking
            tree = ast.parse("".join(self.source_lines)) # No need for type_comments=True for this chunking
            
            # Visit only top-level functions and classes initially.
            # visit_ClassDef will then handle methods inside classes.
            for node in tree.body:
                 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                     self.visit(node)
            logger.info(f"PythonCodeChunker: Finished AST parsing for '{self.file_identifier_for_parent}'. Found {len(self.chunks)} chunks.")
            return self.chunks
        except SyntaxError as e:
             logger.error(f"AST Parsing SyntaxError in '{self.file_identifier_for_parent}': {e}. Cannot chunk accurately.", exc_info=False)
             # Fallback: return whole file as one chunk
             return [{
                 "content": "".join(self.source_lines),
                 "metadata": {
                     "type": "python_file_parse_error", 
                     "identifier": self.file_identifier_for_parent, # Use the base name
                     "sequence_index": 0, 
                     "parent_identifier": None, # No parent for the whole file
                     "start_line": 1,
                     "end_line": len(self.source_lines)
                    }
             }]
        except Exception as e: # Catch other potential AST errors
            logger.error(f"Unexpected error during Python code chunking for '{self.file_identifier_for_parent}': {e}", exc_info=True)
            return [{
                 "content": "".join(self.source_lines),
                 "metadata": {
                     "type": "python_file_chunking_error",
                     "identifier": self.file_identifier_for_parent,
                     "sequence_index": 0,
                     "parent_identifier": None,
                     "start_line": 1,
                     "end_line": len(self.source_lines)
                    }
            }]


# --- Markdown Chunking ---

def chunk_markdown(text: str, file_identifier: str = "unknown_markdown.md") -> List[Dict[str, Any]]:
    logger.info(f"Markdown Chunker: Processing '{file_identifier}'")
    base_file_identifier = os.path.splitext(file_identifier)[0]
    chunks = []
    
    # Split by level 2 headers (##). Level 1 (#) could be document title.
    # Regex captures the header line itself and the content following it.
    # Using lookahead to keep the delimiter (header line) with the content it belongs to.
    parts = re.split(r'(\n##\s+.*?(?=\n##\s+|\Z))', text, flags=re.S)
    
    current_content_accumulator = []
    current_header = "Introduction" # Default for content before first ##
    sequence_counter = 0

    # Handle content before the first "## " header if any
    if parts and parts[0].strip():
        # If the first part does not start with "## ", it's intro content
        if not parts[0].strip().startswith("## "):
            current_content_accumulator.append(parts[0].strip())
        # else: first part is already a ## section, will be handled by loop

    # Iterate through parts, combining header and its content
    idx = 0
    if not parts[0].strip().startswith("## ") and parts[0].strip(): # If intro content exists
        idx = 1 # Start loop from the first captured header

    while idx < len(parts):
        # Save previous section
        if current_content_accumulator:
            chunks.append({
                "content": "\n".join(current_content_accumulator).strip(),
                "metadata": {
                    "type": "markdown_section",
                    "identifier": current_header,
                    "parent_identifier": base_file_identifier,
                    "sequence_index": sequence_counter
                }
            })
            sequence_counter += 1
            current_content_accumulator = []

        # Current part is a header line (e.g., "\n## My Header")
        header_line_full = parts[idx].strip()
        current_header = re.sub(r'^##\s*', '', header_line_full).strip() # Extract header text
        
        # Next part is the content for this header
        idx += 1
        if idx < len(parts):
            content_for_this_header = parts[idx].strip()
            current_content_accumulator.append(content_for_this_header)
        # If no more content parts, this header might be the last thing or formatting issue
        idx +=1


    # Add the last accumulated section
    if current_content_accumulator:
        chunks.append({
            "content": "\n".join(current_content_accumulator).strip(),
            "metadata": {
                "type": "markdown_section",
                "identifier": current_header, # Header of the last section
                "parent_identifier": base_file_identifier,
                "sequence_index": sequence_counter
            }
        })
        sequence_counter += 1

    # If no "##" headers were found, treat the whole file as one chunk
    if not chunks and text.strip():
         chunks.append({
             "content": text.strip(),
             "metadata": {
                 "type": "markdown_file", # Different type for whole-file
                 "identifier": base_file_identifier, # The file itself is the identifier
                 "parent_identifier": None, # No parent for the whole file
                 "sequence_index": 0
                }
         })

    logger.info(f"Chunked Markdown '{file_identifier}' into {len(chunks)} sections.")
    return chunks


# --- Plain Text Chunking ---

def chunk_plain_text(text: str, file_identifier: str = "unknown_text.txt", max_chars: int = 2000) -> List[Dict[str, Any]]:
    logger.info(f"Plain Text Chunker: Processing '{file_identifier}' with max_chars={max_chars}")
    base_file_identifier = os.path.splitext(file_identifier)[0]
    chunks = []
    
    # Split by double newlines (common paragraph separator)
    # Keep empty strings from split to handle multiple blank lines, then filter
    paragraphs = re.split(r'(\n\s*\n)', text) # Capture separators to maintain structure somewhat
    
    current_paragraph_content = ""
    sequence_counter = 0

    full_text_char_counter = 0 # For line number approximation

    for i, part in enumerate(paragraphs):
        if i % 2 == 0: # Content part
            current_paragraph_content += part
        else: # Separator part, means a paragraph ended
            if current_paragraph_content.strip():
                # Now, check if this accumulated paragraph exceeds max_chars
                if len(current_paragraph_content) > max_chars:
                    # Sub-chunk the large paragraph
                    start = 0
                    para_chunk_counter = 1
                    original_paragraph_identifier = f"Paragraph_{sequence_counter + 1}"
                    while start < len(current_paragraph_content):
                        end = start + max_chars
                        # Try to split at a sentence boundary or word boundary near max_chars
                        split_pos = -1
                        # Prefer sentence boundaries
                        for boundary_char in ['.', '!', '?']:
                            temp_split_pos = current_paragraph_content.rfind(boundary_char, start, end)
                            if temp_split_pos != -1:
                                split_pos = max(split_pos, temp_split_pos + 1) # Include the boundary char
                        
                        if split_pos == -1: # No sentence boundary, try word boundary
                            temp_split_pos = current_paragraph_content.rfind(' ', start, end)
                            if temp_split_pos != -1 and end < len(current_paragraph_content): # Ensure it's not the end of the string
                                split_pos = temp_split_pos + 1 # Include space, then strip
                            elif end >= len(current_paragraph_content): # Reached end of paragraph
                                split_pos = len(current_paragraph_content)
                            else: # Cannot find space, hard cut
                                split_pos = end

                        chunk_content = current_paragraph_content[start:split_pos].strip()
                        if chunk_content:
                            chunks.append({
                                "content": chunk_content,
                                "metadata": {
                                    "type": "text_split_paragraph",
                                    "identifier": f"{original_paragraph_identifier}_Part_{para_chunk_counter}",
                                    "parent_identifier": base_file_identifier, # File is parent
                                    "sequence_index": sequence_counter,
                                    # Line numbers are hard for plain text without more context
                                }
                            })
                            sequence_counter += 1
                            para_chunk_counter += 1
                        start = split_pos
                else: # Paragraph is within size limit
                    chunks.append({
                        "content": current_paragraph_content.strip(),
                        "metadata": {
                            "type": "text_paragraph",
                            "identifier": f"Paragraph_{sequence_counter + 1}",
                            "parent_identifier": base_file_identifier,
                            "sequence_index": sequence_counter
                        }
                    })
                    sequence_counter += 1
            current_paragraph_content = "" # Reset for next paragraph
        
        full_text_char_counter += len(part)

    # Add any remaining content (last paragraph)
    if current_paragraph_content.strip():
        # Repeat the logic for splitting if the last paragraph is too large
        if len(current_paragraph_content) > max_chars:
            start = 0
            para_chunk_counter = 1
            original_paragraph_identifier = f"Paragraph_{sequence_counter + 1}"
            while start < len(current_paragraph_content):
                # (Same sub-chunking logic as above)
                end = start + max_chars
                split_pos = -1
                for boundary_char in ['.','!','?']: temp_split_pos = current_paragraph_content.rfind(boundary_char, start, end); split_pos=max(split_pos,temp_split_pos+1) if temp_split_pos!=-1 else split_pos
                if split_pos == -1: temp_split_pos = current_paragraph_content.rfind(' ', start, end); split_pos = temp_split_pos+1 if temp_split_pos!=-1 and end < len(current_paragraph_content) else (len(current_paragraph_content) if end >= len(current_paragraph_content) else end)
                chunk_content = current_paragraph_content[start:split_pos].strip()
                if chunk_content:
                    chunks.append({"content": chunk_content, "metadata": {"type": "text_split_paragraph", "identifier": f"{original_paragraph_identifier}_Part_{para_chunk_counter}", "parent_identifier": base_file_identifier, "sequence_index": sequence_counter}})
                    sequence_counter += 1; para_chunk_counter += 1
                start = split_pos
        else:
            chunks.append({
                "content": current_paragraph_content.strip(),
                "metadata": {
                    "type": "text_paragraph",
                    "identifier": f"Paragraph_{sequence_counter + 1}",
                    "parent_identifier": base_file_identifier,
                    "sequence_index": sequence_counter
                }
            })
            sequence_counter += 1
            
    # If no paragraphs were found (e.g. single line text, or no double newlines)
    if not chunks and text.strip():
        # Treat the whole text as one chunk, but split if it exceeds max_chars
        if len(text) > max_chars:
            start = 0
            part_counter = 1
            while start < len(text):
                # (Same sub-chunking logic as above)
                end = start + max_chars
                split_pos = -1
                for boundary_char in ['.','!','?']: temp_split_pos = text.rfind(boundary_char, start, end); split_pos=max(split_pos,temp_split_pos+1) if temp_split_pos!=-1 else split_pos
                if split_pos == -1: temp_split_pos = text.rfind(' ', start, end); split_pos = temp_split_pos+1 if temp_split_pos!=-1 and end < len(text) else (len(text) if end >= len(text) else end)
                chunk_content = text[start:split_pos].strip()
                if chunk_content:
                    chunks.append({"content": chunk_content, "metadata": {"type": "text_split_content", "identifier": f"{base_file_identifier}_Part_{part_counter}", "parent_identifier": base_file_identifier, "sequence_index": sequence_counter}})
                    sequence_counter +=1; part_counter +=1
                start = split_pos
        else: # Text is small enough for one chunk
            chunks.append({
                "content": text.strip(),
                "metadata": {
                    "type": "text_file", # Whole file as one chunk
                    "identifier": base_file_identifier,
                    "parent_identifier": None,
                    "sequence_index": 0
                }
            })

    logger.info(f"Chunked Plain Text '{file_identifier}' into {len(chunks)} paragraphs/splits.")
    return chunks


# --- Main Chunker Dispatch Function ---

def chunk_file(file_path: str, file_content: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Detects file type from file_path and applies the appropriate chunking strategy.
    If file_content is provided, it's used directly; otherwise, the file is read.
    """
    filename = os.path.basename(file_path)
    # Use filename from path for identifier, not a default like "unknown_file.py"
    
    if file_content is None:
        logger.info(f"Chunker: Reading content from file path: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
        except Exception as e:
            logger.error(f"Chunker: Error reading file {file_path}: {e}", exc_info=True)
            # Return a single error chunk
            return [{
                "content": f"Error reading file: {e}",
                "metadata": {
                    "type": "file_read_error",
                    "identifier": filename,
                    "sequence_index": 0,
                    "parent_identifier": None
                }
            }]
    
    if not file_content.strip():
        logger.warning(f"Chunker: File '{filename}' is empty or contains only whitespace. No chunks generated.")
        return []

    _, extension = os.path.splitext(filename)
    extension = extension.lower()
    logger.info(f"Chunker dispatching for file: {filename} (detected type: {extension})")

    if extension == ".py":
        # Pass the original filename to PythonCodeChunker for its internal identifier
        chunker_instance = PythonCodeChunker(file_content, file_identifier=filename)
        return chunker_instance.chunk()
    elif extension == ".md":
        return chunk_markdown(file_content, file_identifier=filename)
    elif extension in [".txt", ".log", ".sh", ".bat", ".csv", ".json", ".xml", ".html", ".css", ".js"] or not extension: # Treat common text-based as plain
         return chunk_plain_text(file_content, file_identifier=filename)
    else:
        logger.warning(f"Chunker: Unknown file type '{extension}' for '{filename}'. Attempting plain text chunking.")
        return chunk_plain_text(file_content, file_identifier=filename)

# API function to chunk source code string (primarily for Python)
def chunk(source_code: str, file_identifier: str = "api_python_input.py") -> List[Dict[str, Any]]:
    """
    Chunks raw source code string, assuming Python syntax by default for direct API calls.
    `file_identifier` helps in naming the chunks.
    """
    logger.info(f"Chunker API: Chunking raw string content with identifier '{file_identifier}'. Assuming Python.")
    # If other languages need to be chunked via this raw string `chunk` function,
    # a `language` parameter could be added to dispatch to different AST parsers or regex.
    # For now, defaults to Python.
    chunker_instance = PythonCodeChunker(source_code, file_identifier=file_identifier)
    return chunker_instance.chunk()


# --- Example Usage (for testing this script directly) ---
if __name__ == '__main__':
    # Setup basic logging for direct script run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - ChunkerScript - %(message)s')
    
    # Dummy test data (replace with actual file reading or larger strings for real tests)
    test_py_content = """
def top_level_func(a, b):
    # A comment
    return a + b

class MyTestClass:
    class_var = 100

    def __init__(self, val):
        self.instance_var = val

    @classmethod
    def cls_method(cls, x):
        return cls.class_var + x

    def instance_method(self, y):
        # Inner workings
        res = self.instance_var * y
        return res

async def async_example():
    pass
"""
    test_md_content = """
# Document Title

This is an introduction.

## Section One Header

Content for section one.
It can span multiple lines.

## Section Two: Another Header

Content for section two.
    - bullet point 1
    - bullet point 2
"""
    test_txt_content = """This is the first paragraph of a plain text document. It's fairly short.

This is the second paragraph. It might be a bit longer to see how it handles things.
It also has multiple lines within the same paragraph.

This is a very long third paragraph designed to test the max_chars splitting logic. It will go on and on, hopefully exceeding the default limit of 2000 characters to see if it correctly splits into multiple sub-chunks. We need more text here to make sure it's long enough. Let's add some repetitive phrases. This is more text. This is even more text. Still going. Almost there. The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs. We are writing a lot of text to make this paragraph super long and ensure that the splitting logic is triggered effectively. This is sentence one. This is sentence two. This is sentence three, which should ideally be a split point if it aligns with max_chars. And a final sentence.
"""

    logger.info("\n--- Testing Python Chunker ---")
    py_chunks = chunk_file("test_example.py", test_py_content)
    logger.info(f"Found {len(py_chunks)} Python chunks.")
    for i, chk in enumerate(py_chunks):
        meta = chk.get('metadata', {})
        logger.info(f"  Chunk {i+1}: Type={meta.get('type')}, ID='{meta.get('identifier')}', Parent='{meta.get('parent_identifier')}', Seq={meta.get('sequence_index')}, Lines {meta.get('start_line')}-{meta.get('end_line')}, Content len: {len(chk.get('content'))}")
    logger.info("-" * 30)

    logger.info("\n--- Testing Markdown Chunker ---")
    md_chunks = chunk_file("test_example.md", test_md_content)
    logger.info(f"Found {len(md_chunks)} Markdown chunks.")
    for i, chk in enumerate(md_chunks):
        meta = chk.get('metadata', {})
        logger.info(f"  Chunk {i+1}: Type={meta.get('type')}, ID='{meta.get('identifier')}', Parent='{meta.get('parent_identifier')}', Seq={meta.get('sequence_index')}, Content len: {len(chk.get('content'))}")
    logger.info("-" * 30)

    logger.info("\n--- Testing Plain Text Chunker ---")
    # Test with smaller max_chars for the example to show splitting
    txt_chunks = chunk_plain_text(test_txt_content, file_identifier="example_plain.txt", max_chars=300)
    logger.info(f"Found {len(txt_chunks)} Text chunks.")
    for i, chk in enumerate(txt_chunks):
         meta = chk.get('metadata', {})
         logger.info(f"  Chunk {i+1}: Type={meta.get('type')}, ID='{meta.get('identifier')}', Parent='{meta.get('parent_identifier')}', Seq={meta.get('sequence_index')}, Content len: {len(chk.get('content'))}")
    logger.info("-" * 30)

    logger.info("\n--- Testing Raw String Python Chunker (API like) ---")
    api_py_chunks = chunk(test_py_content, file_identifier="api_test_script.py")
    logger.info(f"Found {len(api_py_chunks)} API Python chunks.")
    for i, chk in enumerate(api_py_chunks):
        meta = chk.get('metadata', {})
        logger.info(f"  Chunk {i+1}: Type={meta.get('type')}, ID='{meta.get('identifier')}', Parent='{meta.get('parent_identifier')}', Seq={meta.get('sequence_index')}, Lines {meta.get('start_line')}-{meta.get('end_line')}, Content len: {len(chk.get('content'))}")
    logger.info("-" * 30)