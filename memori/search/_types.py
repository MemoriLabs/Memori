from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bson import ObjectId

    FactId = int | ObjectId
else:
    FactId = int | object  # object allows ObjectId at runtime


@dataclass(frozen=True)
class FactCandidate:
    id: FactId
    content: str
    score: float
    date_created: str


@dataclass(frozen=True)
class FactSearchResult:
    id: FactId
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
