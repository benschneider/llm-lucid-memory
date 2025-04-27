from lucid_memory.memory_node import MemoryNode
from lucid_memory.memory_graph import MemoryGraph

def test_add_and_get_node():
    graph = MemoryGraph()
    node = MemoryNode(
        id="test_node",
        raw="print('Hello World')",
        summary="Prints Hello World.",
        reasoning_paths=["Outputs text to console."],
        tags=["example"],
    )
    graph.add_node(node)
    retrieved = graph.get_node("test_node")
    assert retrieved is not None
    assert retrieved.id == "test_node"
    assert "Hello World" in retrieved.raw

def test_search_by_tag():
    graph = MemoryGraph()
    node1 = MemoryNode(id="node1", raw="a", summary="a", reasoning_paths=["a"], tags=["tag1"])
    node2 = MemoryNode(id="node2", raw="b", summary="b", reasoning_paths=["b"], tags=["tag2"])
    graph.add_node(node1)
    graph.add_node(node2)
    results = graph.search_by_tag("tag1")
    assert len(results) == 1
    assert results[0].id == "node1"

def test_save_and_load(tmp_path):
    graph = MemoryGraph()
    node = MemoryNode(id="node", raw="r", summary="s", reasoning_paths=["r"], tags=["t"])
    graph.add_node(node)
    save_path = tmp_path / "memory.json"
    graph.save_to_json(str(save_path))

    new_graph = MemoryGraph()
    new_graph.load_from_json(str(save_path))
    retrieved = new_graph.get_node("node")
    assert retrieved is not None
    assert retrieved.summary == "s"