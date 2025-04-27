[![License](https://img.shields.io/github/license/benschneider/llm-lucid-memory)](LICENSE)

[![Advanced Reasoning Project](https://img.shields.io/badge/advanced%20reasoning-in%20progress-blue)]()

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
User Question -> ReflectiveRetriever -> Select Relevant Memories -> ChainOfDraftEngine -> Monte Carlo Chain Voting -> Logical Answer

```



## 🧪 Quick Start

Install dependencies:
```bash
pip install -r requirements.txt
```

Start a UI to configure and start a Proxy Server for LLM interaction:
```bash 
python start.py
```

### For development:

Run tests: 
```bash 
pytest 
```

Run a simple ingestion demo: 
```bash 
python -m examples.simple_digest_demo ```
```

## 🧠 Example Usage

Question:

“How does the server start and accept secure connections?”

Memory Retrieval:
	- Memory 1: Start server (open socket, bind port, accept requests)
	- Memory 2: Handle TLS (perform handshake, secure channel)

Drafted Answer:
	- Open socket
	- Bind port
	- Accept connection
	- Initiate TLS handshake
	- Proceed with secured HTTP handling


## 🌱 What's Coming Next?

| Feature                               | Status    |
|:--------------------------------------|:----------|
| ChainOfDraftEngine (parallel reasoning chains) | 🔜 Planned |
| Monte Carlo Chain Voting for Reliable Reasoning | 🔜 Research & Planned |
| MemoryNode versioning                 | 🔜 Future  |
| Graph-based retrieval paths           | 🔜 Future  |
| Sleep-time memory growth (dream ingestion) | 🔜 Future  |
| Fine-tuning LLMs for memory reasoning  | 🔜 Future  |

## 🧠 Advanced Research Directions

### Monte Carlo Chain Voting for Reliable Reasoning

Inspired by the research ["More Reliable Code Generation via Monte Carlo Tree Search"](https://openreview.net/pdf?id=xoXn62FzD0) from MIT (summarized [here](https://news.mit.edu/2025/making-ai-generated-code-more-accurate-0418)), we plan to extend Lucid Memory with a **Monte Carlo Chain Voting** mechanism.

Instead of taking the first drafted reasoning chain, multiple parallel drafts will be generated using the memory graph.  
Chains will be scored for consistency, logical flow, and alignment with memories.  
The strongest chain will be selected or composed as the final answer.

This approach will drastically reduce hallucination, improve reasoning reliability, and enhance long-term memory growth.

Stay tuned as we integrate these innovations into future versions! 🚀

## 📜 License

Apache 2.0 — free for commercial and research use with attribution.

## ✨ Vision

We are building a modular reasoning brain for LLMs:
    - Digest structured memories
    - Reflectively retrieve knowledge
    - Reason flexibly beyond context limits
    - Grow smarter over time

Helping small models think big - the way real minds do. 🚀

## 🔌 How to Use with Local LLMs (Ollama, LMStudio, etc.)

- Start your local LLM server (e.g., Ollama)
- Open `start.py` and configure:
  - Backend URL (e.g., http://localhost:11434/v1/chat/completions)
  - Model Name (e.g., mistral, phi4)
- Launch Proxy Server from the UI
- Point your client to `http://localhost:{your_proxy_port}/chat`
- Enjoy memory-enhanced, chain-of-draft smart reasoning!

## 🚧 Current Status and Next Steps

Lucid Memory is actively under development.

**Current Highlights:**
- Core MemoryNode and MemoryGraph structures are functional and fully tested.
- ReflectiveRetriever operational for smart memory selection.
- Proxy server connects to local LLMs like Ollama and LMStudio, injecting smart memory context.
- First end-to-end smart chain-of-draft flow tested successfully.

**Next Major Implementations:**
- Launch ChainOfDraftEngine to structure logical answer chaining.
- Add Monte Carlo Chain Voting to improve answer reliability and reduce hallucinations.
- Enhance start.py UI to allow manual preprocessing of knowledge (digesting custom text/code snippets).
- Develop a lightweight fine-tuning dataset for memory-enhanced reasoning.

Stay tuned — the brain is growing! 🧠🚀


## 📦 Project Structure

```plaintext
llm-lucid-memory/
├── README.md            # Project overview
├── LICENSE              # Apache 2.0 License
├── lucid_memory/        # Core modules
│   ├── __init__.py
│   ├── digestor.py       # Digest raw input into MemoryNodes
│   ├── memory_node.py    # Single knowledge atoms
│   ├── memory_graph.py   # In-memory brain managing MemoryNodes
│   ├── retriever.py      # ReflectiveRetriever (smart retrieval)
│   ├── proxy_server.py   # Proxy server bridging app ↔ smart LLM context
│   └── start.py          # UI launcher for proxy server and config
├── examples/            # Demos and pipelines
│   ├── simple_digest_demo.py
│   ├── full_pipeline_demo.py
│   └── test_client.py
├── tests/               # Test suites for all modules
│   ├── test_digestor.py
│   ├── test_memory_graph.py
│   ├── test_memory_node.py
│   ├── test_retriever.py
│   └── (more planned)
├── requirements.txt     # Lightweight dependencies
└── setup.py             # Optional pip packaging
```