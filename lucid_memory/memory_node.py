# lucid_memory/memory_node.py

from typing import List, Optional, Dict, Any # Added Dict, Any

class MemoryNode:
    def __init__(self,
                 id: str,
                 raw: str,
                 summary: str,
                 # Keep as key_concepts for now, applies generally
                 key_concepts: List[str],
                 tags: List[str],
                 follow_up_questions: Optional[List[str]] = None,
                 # --- New Linking Fields ---
                 sequence_index: Optional[int] = None,
                 parent_identifier: Optional[str] = None,
                 # --- End New Linking Fields ---
                 source: Optional[str] = None,
                 # --- Add placeholder for code-specific fields ---
                 # Initialize code-specific fields to None or empty lists
                 # to avoid errors if created from non-code chunks initially.
                 inputs: Optional[List[str]] = None,
                 outputs: Optional[List[str]] = None,
                 internal_vars: Optional[List[str]] = None
                 ):
        self.id = id
        self.raw = raw
        self.summary = summary
        self.key_concepts = key_concepts # Keep this name for now
        self.tags = tags
        self.follow_up_questions = follow_up_questions if follow_up_questions is not None else []
        # --- Assign Linking Fields ---
        self.sequence_index = sequence_index
        self.parent_identifier = parent_identifier
        # --- End Assignment ---
        self.source = source # Original source file path (optional)
        # --- Assign Code Fields ---
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if outputs is not None else []
        self.internal_vars = internal_vars if internal_vars is not None else []


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node to a dictionary."""
        return {
            "id": self.id,
            "raw": self.raw,
            "summary": self.summary,
            "key_concepts": self.key_concepts,
            "tags": self.tags,
            "follow_up_questions": self.follow_up_questions,
            # --- Serialize Linking Fields ---
            "sequence_index": self.sequence_index,
            "parent_identifier": self.parent_identifier,
            # --- End Serialization ---
            "source": self.source,
            # --- Serialize Code Fields ---
            "inputs": self.inputs,
            "outputs": self.outputs,
            "internal_vars": self.internal_vars,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Deserializes a node from a dictionary."""
        # Handle potential legacy data without key_concepts
        concepts = data.get("key_concepts", data.get("reasoning_paths", []))

        return cls(
            id=data["id"],
            raw=data.get("raw", ""), # Default to empty string if missing
            summary=data.get("summary", ""), # Default
            key_concepts=concepts,
            tags=data.get("tags", []),
            follow_up_questions=data.get("follow_up_questions", []),
            # --- Deserialize Linking Fields ---
            sequence_index=data.get("sequence_index"), # Defaults to None if missing
            parent_identifier=data.get("parent_identifier"), # Defaults to None
            # --- End Deserialization ---
            source=data.get("source"),
             # --- Deserialize Code Fields ---
            inputs=data.get("inputs", []), # Default to empty list
            outputs=data.get("outputs", []), # Default to empty list
            internal_vars=data.get("internal_vars", []) # Default to empty list
        )

    def __repr__(self):
        """Provides a concise string representation for debugging."""
        # Include new fields in repr
        parent_info = f", parent='{self.parent_identifier}'" if self.parent_identifier else ""
        seq_info = f", seq={self.sequence_index}" if self.sequence_index is not None else ""
        return (f"MemoryNode(id='{self.id}'{seq_info}{parent_info}, "
                f"concepts={len(self.key_concepts)}, tags={len(self.tags)})")