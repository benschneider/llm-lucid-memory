[![Tests](https://img.shields.io/github/workflow/status/benschneider/llm-lucid-memory/Tests)]()

[![License](https://img.shields.io/github/license/benschneider/llm-lucid-memory)](LICENSE)

# 🧠 llm-lucid-memory

**Lucid Memory** is an open-source project aiming to enable small and medium LLMs to **reason beyond their context windows** — through modular memory digestion, structured storage, reflective retrieval, and chain-of-draft reasoning.

> **Imagine:** Your model thinking more like a brain, not just predicting the next token.

---

## 🌟 Why Lucid Memory?

- **Digest** knowledge offline into modular, logic-rich memory nodes
- **Store** memories flexibly (searchable by tags, keywords, and logic paths)
- **Reflectively retrieve** only relevant memories for a question
- **Draft answers** based on structured reasoning, not blind retrieval
- **Unlock huge knowledge bases** even with limited context models

---

## 📚 Concept Overview

**The Problem:**  
- Small LLMs can't fit large codebases or document sets.
- Existing RAG (retrieval augmented generation) just crams context into prompts.
- LLMs need *structured*, *logical*, *reflective* memory systems.

**The Solution:**  
Lucid Memory introduces a lightweight brain architecture for LLMs:
- Preprocess (digest) large information offline.
- Save modular memories with summaries and reasoning paths.
- Reflectively retrieve and reason using only what matters.

---

## 🔥 Core Reasoning Flow

```plaintext
[DIGEST PHASE - offline]
Raw Knowledge -> Digestor -> MemoryNode -> MemoryGraph (saved)

[QUERY PHASE - online]
User Question -> ReflectiveRetriever -> Select Relevant Memories -> ChainOfDraftEngine -> Logical Answer



```



## 🧪 Quick Start

Install dependencies: ``` pip install -r requirements.txt ```

Run tests: ``` pytest ```

Run a simple ingestion demo: ``` python -m examples.simple_digest_demo ```


## 🧠 Example Usage

Question:

“How does the server start and accept secure connections?”

Memory Retrieval:
	•	Memory 1: Start server (open socket, bind port, accept requests)
	•	Memory 2: Handle TLS (perform handshake, secure channel)

Drafted Answer:
	•	Open socket
	•	Bind port
	•	Accept connection
	•	Initiate TLS handshake
	•	Proceed with secured HTTP handling


## 🌱 What's Coming Next?

| Feature                               | Status    |
|:--------------------------------------|:----------|
| ChainOfDraftEngine (parallel reasoning chains) | 🔜 Planned |
| MemoryNode versioning                 | 🔜 Future  |
| Graph-based retrieval paths           | 🔜 Future  |
| Sleep-time memory growth (dream ingestion) | 🔜 Future  |
| Fine-tuning LLMs for memory reasoning  | 🔜 Future  |

## 📜 License

Apache 2.0 — free for commercial and research use with attribution.

## ✨ Vision

We are building a modular reasoning brain for LLMs:
	•	Digest structured memories
	•	Reflectively retrieve knowledge
	•	Reason flexibly beyond context limits
	•	Grow smarter over time

Helping small models think big — the way real minds do. 🚀

## Project Structure

llm-lucid-memory/
├── README.md          # Project overview
├── LICENSE            # Apache 2.0 License
├── lucid_memory/      # Core modules
│   ├── __init__.py
│   ├── digestor.py     # Digest raw input into MemoryNodes
│   ├── memory_node.py  # Single knowledge atoms
│   ├── memory_graph.py # In-memory brain managing MemoryNodes
│   ├── retriever.py    # ReflectiveRetriever (smart retrieval)
│   └── chain_engine.py # ChainOfDraftEngine (planned)
├── examples/          # Demos and pipelines
│   └── simple_digest_demo.py
├── tests/             # Test suites for all modules
│   ├── test_digestor.py
│   ├── test_memory_graph.py
│   ├── test_memory_node.py
│   └── test_retriever.py
├── requirements.txt   # Lightweight requirements
└── setup.py           # Optional pip packaging


