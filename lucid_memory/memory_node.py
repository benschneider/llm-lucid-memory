from typing import List, Optional, Dict, Any
import logging
import time

class MemoryNode:
    def __init__(self,
                 id: str,
                 raw: str,                                
                 summary: str,                            
                 key_concepts: List[str],                 
                 tags: List[str],                         
                 sequence_index: Optional[int] = None,    
                 parent_identifier: Optional[str] = None, 
                 dependencies: Optional[List[str]] = None,    
                 produced_outputs: Optional[List[str]] = None,
                 follow_up_questions: Optional[List[str]] = None,
                 source: Optional[str] = None,            
                 embedding: Optional[List[float]] = None
                 ):
        if not id: # Ensure ID is always present
            logging.warning("MemoryNode created with empty ID. This is problematic.")
            self.id = f"invalid_node_{int(time.time())}" # Placeholder
        else:
            self.id = id
            
        self.raw = raw
        self.summary = summary
        self.key_concepts = key_concepts        
        self.tags = tags
        self.sequence_index = sequence_index
        self.parent_identifier = parent_identifier
        self.dependencies = dependencies if dependencies is not None else []
        self.produced_outputs = produced_outputs if produced_outputs is not None else []
        self.follow_up_questions = follow_up_questions if follow_up_questions is not None else []
        self.source = source
        self.embedding = embedding

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "raw": self.raw,
            "summary": self.summary,
            "key_concepts": self.key_concepts,
            "tags": self.tags,
            "sequence_index": self.sequence_index,
            "parent_identifier": self.parent_identifier,
            "dependencies": self.dependencies,
            "produced_outputs": self.produced_outputs,
            "follow_up_questions": self.follow_up_questions,
            "source": self.source,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        node_id = data.get("id", "") # Handle missing ID
        if not node_id:
             logging.warning(f"MemoryNode.from_dict: 'id' field is missing or empty in source data: {str(data)[:100]}... Assigning fallback ID.")
             fallback_id_hint = data.get("summary", data.get("raw",""))[:30].replace(" ","_")
             node_id = f"deserialized_node_missing_id_{fallback_id_hint}_{int(time.time())%10000}"


        concepts = data.get("key_concepts", data.get("logical_steps", data.get("reasoning_paths", [])))
        if not isinstance(concepts, list): # Ensure it's a list
            logging.warning(f"Node {node_id}: 'key_concepts' field was not a list (type: {type(concepts)}), defaulting to empty list.")
            concepts = []
        
        embedding_data = data.get("embedding")
        if embedding_data is not None and not isinstance(embedding_data, list):
            logging.warning(f"Node {node_id}: 'embedding' field was not a list (type: {type(embedding_data)}), setting to None.")
            embedding_data = None
        elif embedding_data and not all(isinstance(x, (float, int)) for x in embedding_data if x is not None): # Allow None in list for robustness? No, should be clean.
            logging.warning(f"Node {node_id}: 'embedding' list contains non-numeric values. Setting to None.")
            embedding_data = None


        return cls(
            id=node_id,
            raw=data.get("raw", ""),
            summary=data.get("summary", ""),
            key_concepts=concepts,
            tags=data.get("tags", []),
            sequence_index=data.get("sequence_index"),
            parent_identifier=data.get("parent_identifier"),
            dependencies=data.get("dependencies", []), 
            produced_outputs=data.get("produced_outputs", []),
            follow_up_questions=data.get("follow_up_questions", []),
            source=data.get("source"),
            embedding=embedding_data # Deserialize embedding
        )

    def __repr__(self):
        parent_info = f", parent='{self.parent_identifier}'" if self.parent_identifier else ""
        seq_info = f", seq={self.sequence_index}" if self.sequence_index is not None else ""
        deps_info = f", deps={len(self.dependencies)}" if self.dependencies else ""
        outs_info = f", outs={len(self.produced_outputs)}" if self.produced_outputs else ""
        embed_info = f", emb_dim={len(self.embedding)}" if self.embedding else ", no_emb"
        return (f"MemoryNode(id='{self.id}'{seq_info}{parent_info}{deps_info}{outs_info}{embed_info}, "
                f"concepts={len(self.key_concepts)}, tags={len(self.tags)})")
