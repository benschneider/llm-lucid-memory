import json
from typing import Dict, List, Optional
from lucid_memory.memory_node import MemoryNode

class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}

    def add_node(self, node: MemoryNode):
        self.nodes[node.id] = node

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        return self.nodes.get(node_id)

    def search_by_tag(self, tag: str) -> List[MemoryNode]:
        return [node for node in self.nodes.values() if tag in node.tags]

    def save_to_json(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({node_id: node.to_dict() for node_id, node in self.nodes.items()}, f, indent=2)

    def load_from_json(self, filepath: str):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.nodes = {node_id: MemoryNode.from_dict(node_dict) for node_id, node_dict in data.items()}