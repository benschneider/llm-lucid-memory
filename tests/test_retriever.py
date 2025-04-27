

from lucid_memory.memory_graph import MemoryGraph
from lucid_memory.memory_node import MemoryNode
from lucid_memory.retriever import ReflectiveRetriever

def test_retrieve_by_tag():
    graph = MemoryGraph()
    node1 = MemoryNode(id="node1", raw="a", summary="Handles network traffic", reasoning_paths=["Accepts connections"], tags=["network"])
    node2 = MemoryNode(id="node2", raw="b", summary="Manages database access", reasoning_paths=["Query database"], tags=["database"])
    graph.add_node(node1)
    graph.add_node(node2)

    retriever = ReflectiveRetriever(graph)
    results = retriever.retrieve_by_tag("network")
    assert len(results) == 1
    assert results[0].id == "node1"

def test_retrieve_by_keyword():
    graph = MemoryGraph()
    node = MemoryNode(id="node", raw="c", summary="Processes HTTP requests", reasoning_paths=["Parse headers", "Route requests"], tags=["server"])
    graph.add_node(node)

    retriever = ReflectiveRetriever(graph)
    results = retriever.retrieve_by_keyword("HTTP")
    assert len(results) == 1
    assert results[0].id == "node"

def test_reflect_on_candidates():
    graph = MemoryGraph()
    node1 = MemoryNode(id="node1", raw="x", summary="Handles server startup", reasoning_paths=["Bind socket"], tags=["server"])
    node2 = MemoryNode(id="node2", raw="y", summary="Manages database", reasoning_paths=["Initialize tables"], tags=["database"])
    graph.add_node(node1)
    graph.add_node(node2)

    retriever = ReflectiveRetriever(graph)
    candidates = graph.nodes.values()
    refined = retriever.reflect_on_candidates(list(candidates), "How does the server start?")
    assert len(refined) >= 1
    assert refined[0].id == "node1"