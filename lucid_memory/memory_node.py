from typing import List, Optional

class MemoryNode:
    def __init__(self, id: str, raw: str, summary: str, reasoning_paths: List[str], tags: List[str], source: Optional[str] = None):
        self.id = id
        self.raw = raw
        self.summary = summary
        self.reasoning_paths = reasoning_paths
        self.tags = tags
        self.source = source

    def to_dict(self):
        return {
            "id": self.id,
            "raw": self.raw,
            "summary": self.summary,
            "reasoning_paths": self.reasoning_paths,
            "tags": self.tags,
            "source": self.source
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            id=data["id"],
            raw=data["raw"],
            summary=data["summary"],
            reasoning_paths=data["reasoning_paths"],
            tags=data["tags"],
            source=data.get("source")
        )