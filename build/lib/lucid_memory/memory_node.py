from typing import List, Optional, Dict, Any

class MemoryNode:
    def __init__(self,
                 id: str,
                 raw: str,                                # Raw source text of the chunk
                 summary: str,                            # LLM-generated concise summary
                 key_concepts: List[str],                 # LLM-generated concepts/logic steps within chunk
                 tags: List[str],                         # LLM-generated keywords
                 # --- Linking Fields ---
                 sequence_index: Optional[int] = None,    # Order within the source file/context
                 parent_identifier: Optional[str] = None, # Identifier of parent (class, file)
                 # --- Generated Dependency/Output Fields ---
                 dependencies: Optional[List[str]] = None,    # LLM-identified external needs (params, vars, func calls?)
                 produced_outputs: Optional[List[str]] = None,# LLM-identified results (returns, modified vars)
                 # --- Optional Fields ---
                 follow_up_questions: Optional[List[str]] = None,
                 source: Optional[str] = None             # Original source file path
                 # --- Placeholders for future fields ---
                 # vector_embedding: Optional[List[float]] = None
                 # relationships: Optional[List[Dict]] = None # e.g., [{"type": "calls", "target_id": "...", score: 0.9}]
                 ):
        self.id = id
        self.raw = raw
        self.summary = summary
        self.key_concepts = key_concepts        # Renaming to logical_steps specifically for code might be good later
        self.tags = tags
        # -- Linking --
        self.sequence_index = sequence_index
        self.parent_identifier = parent_identifier
        # -- Dependencies / Outputs --
        # Ensure these are always lists, even if None is passed
        self.dependencies = dependencies if dependencies is not None else []
        self.produced_outputs = produced_outputs if produced_outputs is not None else []
        # -- Optional --
        self.follow_up_questions = follow_up_questions if follow_up_questions is not None else []
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the node to a dictionary."""
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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNode':
        """Deserializes a node from a dictionary."""
        # Handle potential legacy data or different digestion strategies
        concepts = data.get("key_concepts", data.get("logical_steps", data.get("reasoning_paths", [])))

        return cls(
            id=data.get("id", "MISSING_ID"), # Add default
            raw=data.get("raw", ""),
            summary=data.get("summary", ""),
            key_concepts=concepts, # Use retrieved concepts/steps
            tags=data.get("tags", []),
            sequence_index=data.get("sequence_index"),
            parent_identifier=data.get("parent_identifier"),
            dependencies=data.get("dependencies", []), # Default to empty list
            produced_outputs=data.get("produced_outputs", []), # Default to empty list
            follow_up_questions=data.get("follow_up_questions", []),
            source=data.get("source")
        )

    def __repr__(self):
        """Provides a concise string representation for debugging."""
        parent_info = f", parent='{self.parent_identifier}'" if self.parent_identifier else ""
        seq_info = f", seq={self.sequence_index}" if self.sequence_index is not None else ""
        deps_info = f", deps={len(self.dependencies)}" if self.dependencies else ""
        outs_info = f", outs={len(self.produced_outputs)}" if self.produced_outputs else ""
        return (f"MemoryNode(id='{self.id}'{seq_info}{parent_info}{deps_info}{outs_info}, "
                f"concepts={len(self.key_concepts)}, tags={len(self.tags)})")