import os
import openai
from lucid_memory.digestor import Digestor
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.retriever import ReflectiveRetriever

# Set your OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def build_prompt(memories, question):
    prompt = "You have the following memories:\n\n"
    for i, memory in enumerate(memories, 1):
        prompt += f"Memory {i}:\nSummary: {memory.summary}\nReasoning Paths:\n"
        for rp in memory.reasoning_paths:
            prompt += f"- {rp}\n"
        prompt += "\n"
    prompt += f"Question:\n{question}\n\n"
    prompt += "Please reason using the memories provided. Draft the steps logically."
    return prompt

def main():
    # Initialize components
    digestor = Digestor()
    graph = MemoryGraph()

    # Digest a few pieces of knowledge
    raw_texts = [
        ("func_start_server", "def start_server(port):\n    # Open socket\n    # Bind port\n    # Accept HTTP requests"),
        ("func_handle_tls", "def handle_tls_connection(conn):\n    # Perform TLS handshake\n    # Secure the communication channel"),
        ("func_database_query", "def query_database(sql):\n    # Connect to DB\n    # Execute SQL\n    # Return results"),
    ]

    for node_id, text in raw_texts:
        node = digestor.digest(text, node_id=node_id)
        graph.add_node(node)

    # Initialize retriever
    retriever = ReflectiveRetriever(graph)

    # User question
    question = "How does the server start and accept secure connections?"

    # Retrieve candidate memories
    candidates = retriever.retrieve_by_keyword("server") + retriever.retrieve_by_keyword("tls")
    candidates = list(set(candidates))
    best_nodes = retriever.reflect_on_candidates(candidates, question)

    # Build prompt
    prompt = build_prompt(best_nodes, question)

    # Query OpenAI
    print("Sending to OpenAI...")
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful reasoning assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
    )

    # Output
    print("\n=== Drafted Answer ===\n")
    print(response["choices"][0]["message"]["content"])

if __name__ == "__main__":
    main()