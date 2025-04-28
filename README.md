[![License](https://img.shields.io/github/license/benschneider/llm-lucid-memory)](LICENSE)
[![PyPI version](https://badge.fury.io/py/llm-lucid-memory.svg)](https://badge.fury.io/py/llm-lucid-memory)
[![Project Status](https://img.shields.io/badge/status-v0.2.1-green)](https://github.com/benschneider/llm-lucid-memory/releases/tag/v0.2.1)

---

## 📚 Table of Contents
- [Vision: The Reasoning Graph](#-vision-the-reasoning-graph)
- [Why Lucid Memory?](#-why-lucid-memory)
- [Core Concept: Graph-Based Reasoning](#-core-concept-graph-based-reasoning)
- [Enhanced Digestion Process](#-enhanced-digestion-process-chunking--pre-processing)
    - [Text Document Digestion](#text-document-digestion)
    - [Code Digestion (Python Example)](#code-digestion-python-example)
- [Core Reasoning Flow](#-core-reasoning-flow)
- [Quick Start](#-quick-start)
- [Example Usage](#-example-usage)
- [v0.3 Roadmap: Building the Graph Foundation](#%EF%B8%8F-v03-roadmap-building-the-graph-foundation)
- [Advanced Research Directions](#-advanced-research-directions)
- [License](#-license)
- [How to Use with Local LLMs](#-how-to-use-with-local-llms-ollama-lmstudio-etc)
- [Current Status and Next Steps (v0.2.x)](#-current-status-and-next-steps-v02x)
- [Project Structure](#-project-structure)

---

# 🧠 llm-lucid-memory

**Lucid Memory** is an open-source project evolving towards a **reasoning graph** for LLMs. It enables smaller models to comprehend and reason about large codebases and document sets by breaking them into understandable chunks, pre-processing the logic within each chunk, and connecting them structurally.

> **Imagine:** Your LLM navigating a graph of interconnected concepts and code logic, not just processing isolated text snippets.

---

## ✨ Vision: The Reasoning Graph

We are building a system that allows LLMs to:
    - **Chunk** large documents and codebases into meaningful, context-aware units (sections, functions, classes).
    - **Digest** each chunk to extract not just summaries, but **pre-processed understanding** (like logical steps in code) using LLMs.
    - **Connect** these digested chunks (Memory Nodes) into a **graph** representing structural and potentially logical relationships.
    - **Retrieve** knowledge not just by keywords or semantic similarity, but by **traversing the graph** to gather relevant, linked context.
    - **Reason** over this structured, interconnected knowledge graph, enabling deeper comprehension and more accurate outputs, even with limited context windows.

Helping small models think big by building structured, navigable "mental models" of information. 🚀

---

## 🌟 Why Lucid Memory?

- **Go Beyond Context Limits:** Overcome the input size limitations of smaller LLMs.
- **Deep Understanding:** Move past simple RAG retrieval towards comprehending structure and logic.
- **Structured Knowledge:** Represent information as an interconnected graph, not just isolated vector chunks.
- **Code Comprehension:** Specifically designed to pre-process and understand the logic within code functions/classes.
- **Efficient Retrieval:** Graph traversal allows targeted retrieval of necessary linked context.
- **Modular & Extensible:** Core components (chunker, digestor, graph, retriever) are designed for flexibility.

---

## 🧩 Core Concept: Graph-Based Reasoning

**The Problem:**
- Small LLMs struggle with large, complex inputs (codebases, documentation).
- Simple RAG often retrieves irrelevant or incomplete context due to lack of structural awareness.
- LLMs need systems that mirror structured thinking and knowledge linking.

**The Lucid Memory Solution:**
1.  **Structure-Aware Chunking:** Break down input based on its type (Markdown headers, code functions/classes via Abstract Syntax Trees).
2.  **Multi-faceted Digestion:** For each chunk, use targeted LLM calls to extract:
    *   Concise Summary
    *   Key Concepts / Logical Steps (Chain-of-Thought for code)
    *   Key Variables / Entities (for code)
    *   Relevant Tags
3.  **Memory Node Creation:** Store digested chunk information in a `MemoryNode`.
4.  **Graph Construction:** Link `MemoryNode`s based on document structure (e.g., sequence, hierarchy) and potentially code calls (future).
5.  **Graph Retrieval:** Find relevant nodes via semantic/keyword search, then traverse the graph to gather connected, contextual information.
6.  **Contextual Generation:** Provide the structured, retrieved graph context to the LLM for informed reasoning and answer generation.

---

## 🔥 Enhanced Digestion Process (Chunking & Pre-processing)

Instead of digesting entire files, Lucid Memory first chunks the input intelligently based on its type. Each chunk is then digested using multiple targeted LLM calls:

### Text Document Digestion (e.g., Markdown)

1.  **Chunking:** Split by semantic sections (e.g., using `##` or `###` headers).
2.  **LLM Calls per Chunk:**
    *   **Summary:** Generate a 1-sentence summary of the section.
    *   **Key Concepts:** Extract core ideas or topics discussed in the section (keywords/short phrases).
    *   **Tags:** Generate relevant keyword tags for the section.
    *   **Follow-up Questions:** Identify ambiguities or areas needing clarification within the section.
3.  **Node Creation:** Create a `MemoryNode` for the section, storing digested info and linking it (e.g., to the previous section).

### Code Digestion (Python Example)

1.  **Chunking:** Use the `ast` module to parse the Python file. Create chunks for each function (`FunctionDef`) or class (`ClassDef`).
2.  **LLM Calls per Code Chunk (Function/Method):**
    *   **Summary:** Generate a 1-sentence summary of the function's purpose (akin to a docstring).
    *   **Logical Steps (CoT):** Extract the high-level conceptual steps the function performs (its internal logic).
    *   **Key Variables:** Identify key input parameters, important local variables, and return values.
    *   **Tags:** Generate relevant technical tags (e.g., `api call`, `database query`, `validation`).
3.  **Node Creation:** Create a `MemoryNode` for the function/class, storing the detailed digested info and linking it to its parent module/class node.

This chunk-based, multi-faceted digestion creates rich, structured nodes ready for graph construction and retrieval.

---

## 🔄 Core Reasoning Flow (Conceptual v0.3+)

```mermaid
graph LR
    A[Raw Knowledge\n(File/Text)] --> B{Structure-Aware\nChunker};
    B --> C(Chunk);
    C --> D{Multi-Call LLM\nDigestor};
    D --> E[Memory Node\n(Summary, Logic,\nVars, Tags)];
    E --> F((Memory Graph\n+ Relationships));

    G[User Question] --> H{Graph Retriever\n(Search + Traversal)};
    F --> H;
    H --> I[Relevant Context Nodes];
    I --> J{ChainOfDraftEngine\n(Planned)};
    G --> J;
    J --> K[Logical Answer];

    style F fill:#f9f,stroke:#333,stroke-width:2px
    style H fill:#ccf,stroke:#333,stroke-width:2px
```

## 🧪 Quick Start

### Dependencies:

```bash
pip install -r requirements.txt
# (Requires: fastapi, uvicorn, requests, PyYAML)
```

#### Configuration:

1. Copy or rename lucid_memory/proxy_config.example.json to lucid_memory/proxy_config.json.
2. Edit lucid_memory/proxy_config.json with your local LLM API endpoint (e.g., Ollama, LMStudio v1 compatible) and the desired model name.
3. (Optional) Edit lucid_memory/prompts.yaml to customize LLM instructions.


### Run the GUI:

Start a UI to configure and start a Proxy Server for LLM interaction:
```bash 
python -m lucid_memory.gui
```

#### From the GUI

1. Verify configuration.
2. Click "Load Context File" to select a file (.md, .py, .txt) for chunking and digestion (can take time).
3. Click "Start Proxy Server".
4. Use the Chat interface to ask questions related to your loaded context.

### For development:

Run tests: 
```bash 
pytest 
```
*(Note: Tests need updating for chunking/new digestion process)*

Run a simple ingestion demo: 
```bash 
python -m examples.simple_digest_demo ```
```
---

## 🧠 Example Usage

**(Conceptual Example with Digested Code)**

**Digested Node (Function Chunk):**
*   **ID:** `file_server_utils_py_func_start_secure_server_1`
*   **Summary:** Starts a server socket, binds to a port, and wraps connections with TLS for secure communication.
*   **Logical Steps:**
    *   Create TCP socket.
    *   Set socket options (e.g., reuse address).
    *   Bind socket to host and port.
    *   Listen for incoming connections.
    *   Load TLS certificate and key.
    *   Create TLS context.
    *   Enter main loop: accept connection.
    *   Wrap accepted socket with TLS context.
    *   Handle client request over TLS (delegate).
*   **Key Variables:** `input: port (int)`, `input: certfile (str)`, `input: keyfile (str)`, `internal: sock (socket)`, `internal: context (SSLContext)`, `internal: conn (socket)`
*   **Tags:** `server`, `socket`, `network`, `tls`, `ssl`, `security`, `listener`

**User Question:**
“How are secure connections handled when the server starts?”

**Graph Retrieval:**
1.  Keyword search finds `file_server_utils_py_func_start_secure_server_1`.
2.  (Future) Graph traversal might pull related nodes if referenced.

**Proxy Prompt Context (Simplified):**

Relevant Memory:
--- Memory 1 (ID: file_server_utils_py_func_start_secure_server_1) ---
Summary: Starts a server socket, binds to a port, and wraps connections with TLS for secure communication.
Key Concepts/Logic:

- Create TCP socket
- Bind socket to host and port
- Listen for incoming connections
- Load TLS certificate and key
- Create TLS context
- Accept connection
- Wrap accepted socket with TLS context
- Handle client request over TLS

Tags: server, socket, network, tls, ssl, security


Based ONLY on the memories provided, answer:
Question: How are secure connections handled when the server starts?


**LLM Drafted Answer:**
The server handles secure connections by: loading a TLS certificate/key, creating a TLS context, accepting incoming connections, and then wrapping the connection socket with the TLS context before handling the client request.


## 🚀 v0.3 Roadmap: Building the Graph Foundation

The focus for v0.3 is implementing the core chunking, digestion, and basic graph structure.

| Feature                                           | Status          | Notes                                                                |
| :------------------------------------------------ | :-------------- | :------------------------------------------------------------------- |
| **Core:** Structure-Aware Chunking                | ✅ In Progress  | Basic MD (headers) & Python (`ast` functions/methods) implemented. |
| **Core:** Implement Basic Node Linking            | 🚀 **Next Up!** | Store `sequence_index`, `parent_identifier` based on chunking.       |
| **Core:** Update `MemoryNode` Fields              | 📝 Planned      | Add linking fields (`sequence_index`, `parent_identifier`).        |
| **Core:** Update `prompts.yaml` (Code Focus)      | 📝 Planned      | Add/Refine prompts for `logical_steps`, `key_variables` (Code).    |
| **Core:** Refactor `Digestor` (Code Focus)        | 📝 Planned      | Add logic to use code-specific prompts based on chunk type.        |
| **Core:** Update `Processor` Logic                | 📝 Planned      | Pass linking info during node creation/storage.                    |
| **Core:** Add `PyYAML` Dependency                 | ✅ Done         | Added to requirements.                                               |
| **Enhancement:** Update `Retriever` (Basic Graph) | 📝 Planned      | Add simple neighbour fetching based on stored links.                 |
| **Enhancement:** Update `ProxyServer` Prompting   | 📝 Planned      | Use `logical_steps`/`key_variables` from nodes in context.         |
| **Future:** Parallel Chunk Digestion            | ✅ Done         | Using `ThreadPoolExecutor` in `Processor`.                         |
| **Future:** Advanced Relationship Extraction      | 💤 Future       | Code call graphs, conceptual links.                              |
| ChainOfDraftEngine                                | 💤 Future       | Postponed - Requires solid graph retrieval first.                  |
| Monte Carlo Chain Voting                          | 💤 Future       | Postponed - Depends on ChainOfDraftEngine.                       |
Stay tuned as we build the foundation for true graph-based reasoning! 🧠🕸️



## 🧠 Advanced Research Directions

### 🧠 Future Optimization: Monte Carlo Chain Voting for Memory Refinement

- **Monte Carlo Tree Search (MCTS) for Reasoning:** Explore MCTS not just for voting on final answers (as planned later) but potentially for *navigating* the memory graph during retrieval to find optimal reasoning paths. (inspired by: ["More Reliable Code Generation via Monte Carlo Tree Search"](https://openreview.net/pdf?id=xoXn62FzD0) from MIT (summarized [here](https://news.mit.edu/2025/making-ai-generated-code-more-accurate-0418)))
- **Graph Optimization:** Use MCCV or other techniques during "sleep time" to analyze the memory graph, identify weak/redundant nodes or paths, and potentially refine summaries or links based on usage patterns.
- **Automated Relationship Extraction:** Develop more sophisticated techniques (LLM-based analysis, advanced static analysis) to automatically infer complex relationships like code call graphs, data flow, or conceptual similarities between nodes.
- **Fine-tuning for Graph Reasoning:** Create datasets and methods specifically for fine-tuning smaller LLMs to better utilize the structured graph context provided by Lucid Memory during generation.


## 📜 License

Apache 2.0 — free for commercial and research use with attribution.

## 🔌 How to Use with Local LLMs (Ollama, LMStudio, etc.)

- Start your local LLM server (ensure it exposes an OpenAI-compatible API endpoint).
- Edit `lucid_memory/proxy_config.json`:
    - Set `backend_url` to your LLM's API endpoint (e.g., `http://localhost:11434/v1/chat/completions`).
    - Set `model_name` to the identifier your local LLM uses.
- Launch the Lucid Memory GUI: `python -m lucid_memory.gui`
- Use the GUI to "Load Context File" (this performs chunking & digestion).
- Start the Proxy Server via the GUI button.
- Chat using the GUI's chat interface. The proxy injects graph-retrieved context.

---

## 🚧 Current Status and Next Steps (v0.2.1)

Lucid Memory v0.2.1 introduces the core chunking and parallel processing pipeline:

**Current Highlights (v0.2.1):**
- **Structure-Aware Chunking:** Basic implementation for Markdown (by `##` headers) and Python (by functions/methods via `ast`) is functional (`chunker.py`).
- **Parallel Digestion:** Chunks are processed in parallel using `ThreadPoolExecutor` for improved performance (`processor.py`).
- **Refactored GUI:** Cleaner separation between UI (`gui.py`) and background processing (`processor.py`).
- **Multi-call `Digestor`:** Uses configurable YAML prompts for robust extraction of summary, concepts, and tags per chunk.
- **Core `MemoryNode`/`MemoryGraph`:** Saving/loading graph with digested chunk info works.
- **Basic `Retriever`/`ProxyServer`:** Functional but not yet graph-aware.

**Focus Towards v0.3:**
The immediate next step is **implementing basic node linking**. This involves:
1.  Adding fields (`sequence_index`, `parent_identifier`) to `MemoryNode`.
2.  Extracting and storing this linking information during the chunking/processing stage.
This will connect the individual chunk nodes, forming the initial graph structure needed for context-aware retrieval – the foundation for v0.3. Following that, refining the code digestion prompts and updating the retriever/proxy will be priorities.

Stay tuned — the graph is taking shape! 🧠🕸️

**Focus for v0.3 (See Roadmap Above):**
The immediate next steps involve implementing the **structure-aware chunking** (Markdown, Python `ast`), updating the **digestion process** to extract richer understanding (logical steps, variables) per chunk, and storing **basic structural links** between nodes. This transitions the project from whole-file analysis to building the foundational **reasoning graph**.

Stay tuned — the brain is evolving into a graph! 🧠🕸️

---

## 📦 Project Structure

```plaintext
llm-lucid-memory/
├── README.md # Project overview (This file)
├── LICENSE # Apache 2.0 License
├── requirements.txt # Project dependencies (fastapi, uvicorn, requests, PyYAML)
├── setup.py # Optional pip packaging
├── lucid_memory/ # Core library source code
│ ├── init.py
│ ├── prompts.yaml # LLM prompt templates
│ ├── proxy_config.json # Runtime configuration for LLM backend / proxy port
│ ├── chunker.py # NEW: Module for structure-aware chunking
│ ├── processor.py # NEW: Handles parallel chunk processing & digestion
│ ├── digestor.py # Digests text chunks using LLM calls (via prompts.yaml)
│ ├── memory_node.py # Definition of a single node in the memory graph
│ ├── memory_graph.py # Manages the collection of MemoryNodes
│ ├── retriever.py # Retrieves nodes (keyword -> graph traversal planned)
│ ├── proxy_server.py # FastAPI proxy intercepting chat, adding context
│ └── gui.py # Tkinter GUI application (Refactored)
├── examples/ # Demo scripts (May need updates for v0.3)
│ ├── simple_digest_demo.py
│ ├── full_pipeline_demo.py
│ └── test_client.py
└── tests/ # Unit and integration tests (Need updates)
├── test_digestor.py
├── test_memory_graph.py
├── test_memory_node.py
├── test_retriever.py
└── (more planned for chunking, processor, etc.)
```