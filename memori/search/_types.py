from dataclasses import dataclass


@dataclass(frozen=True)
class FactCandidate:
    id: int
    content: str
    score: float
    date_created: str


@dataclass(frozen=True)
class FactSearchResult:
    id: int
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
