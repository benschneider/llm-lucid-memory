

from lucid_memory.digestor import Digestor
from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.retriever import ReflectiveRetriever

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

    # Simulate a user question
    question = "How does the server start and accept secure connections?"

    # Retrieve candidate memories
    candidates = retriever.retrieve_by_keyword("server") + retriever.retrieve_by_keyword("tls")
    candidates = list(set(candidates))  # Deduplicate

    # Reflect and rank candidates
    best_nodes = retriever.reflect_on_candidates(candidates, question)

    # Show results
    print(f"Question: {question}\n")
    for node in best_nodes:
        print(f"Memory Node: {node.id}")
        print(f"Summary: {node.summary}")
        print(f"Reasoning Paths: {node.reasoning_paths}")
        print()

if __name__ == "__main__":
    main()