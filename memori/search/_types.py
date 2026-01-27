from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FactCandidate:
    id: Any  # int for SQL, ObjectId for MongoDB
    content: str
    score: float
    date_created: str


@dataclass(frozen=True)
class FactSearchResult:
    id: Any  # int for SQL, ObjectId for MongoDB
    content: str
    similarity: float
    rank_score: float
    date_created: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "content": self.content,
            "similarity": self.similarity,
            "rank_score": self.rank_score,
            "date_created": self.date_created,
        }
