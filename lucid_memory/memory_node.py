from typing import List, Optional

class MemoryNode:
    def __init__(self,
                 id: str,
                 raw: str,
                 summary: str,
                 key_concepts: List[str], # Renamed from reasoning_paths
                 tags: List[str],
                 follow_up_questions: Optional[List[str]] = None,
                 source: Optional[str] = None):
        self.id = id
        self.raw = raw
        self.summary = summary
        self.key_concepts = key_concepts # Renamed
        self.tags = tags
        self.follow_up_questions = follow_up_questions if follow_up_questions is not None else []
        self.source = source

    def to_dict(self):
        return {
            "id": self.id,
            "raw": self.raw,
            "summary": self.summary,
            "key_concepts": self.key_concepts, # Renamed
            "tags": self.tags,
            "follow_up_questions": self.follow_up_questions,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data):
        # Get key_concepts, but fall back to reasoning_paths if loading old data
        concepts = data.get("key_concepts", data.get("reasoning_paths", []))
        return cls(
            id=data["id"],
            raw=data["raw"],
            summary=data["summary"],
            key_concepts=concepts, # Use the potentially retrieved value
            tags=data.get("tags", []),
            follow_up_questions=data.get("follow_up_questions", []),
            source=data.get("source")
        )

    def __repr__(self):
        return (f"MemoryNode(id='{self.id}', summary='{self.summary[:50]}...', "
                f"concepts={len(self.key_concepts)}, tags={len(self.tags)}, " # Renamed
                f"questions={len(self.follow_up_questions)})")