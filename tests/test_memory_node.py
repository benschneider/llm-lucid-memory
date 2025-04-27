from lucid_memory.memory_node import MemoryNode

def test_memory_node_serialization():
    node = MemoryNode(
        id="test_node",
        raw="print('Hello World')",
        summary="Simple print statement.",
        reasoning_paths=["Outputs text to console."],
        tags=["python", "print"],
        source="examples/hello.py"
    )
    node_dict = node.to_dict()
    recovered_node = MemoryNode.from_dict(node_dict)
    
    assert recovered_node.id == node.id
    assert recovered_node.raw == node.raw
    assert recovered_node.summary == node.summary
    assert recovered_node.reasoning_paths == node.reasoning_paths
    assert recovered_node.tags == node.tags
    assert recovered_node.source == node.source