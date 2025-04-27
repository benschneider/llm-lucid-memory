from typing import List, Optional
from lucid_memory.memory_node import MemoryNode

class Digestor:
    def __init__(self):
        pass

    def digest(self, raw_text: str, node_id: str):
        # Extract reasoning paths: look for lines with comments
        reasoning_paths = []
        for line in raw_text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                reasoning_paths.append(line.lstrip("#").strip())
            elif "#" in line:
                reasoning_paths.append(line.split("#", 1)[-1].strip())

        if not reasoning_paths:
            reasoning_paths = ["General description of the function."]

        return MemoryNode(
            id=node_id,
            raw=raw_text,
            summary=raw_text.splitlines()[0],  # For now, first line as rough summary
            reasoning_paths=reasoning_paths,
            tags=self.auto_tag(raw_text)
    )

    def _summarize(self, text: str) -> str:
        return text.split(".")[0] if "." in text else text

    def _extract_reasoning_paths(self, text: str) -> List[str]:
        return [line.strip() for line in text.split(".") if line.strip()]

    def auto_tag(self, text: str) -> List[str]:
        keywords = []
        if "server" in text.lower():
            keywords.append("server")
        if "memory" in text.lower():
            keywords.append("memory")
        if "network" in text.lower():
            keywords.append("network")
        return keywords
