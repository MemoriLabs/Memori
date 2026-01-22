from collections.abc import Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class FactCandidate:
    id: str
    content: str
    score: float


@dataclass(frozen=True)
class FactCandidates:
    facts: list[FactCandidate]


@dataclass(frozen=True)
class FactSearchResult(Mapping[str, object]):
    id: int | str
    content: str
    similarity: float
    rank_score: float
    date_created: str | None = None

    def __getitem__(self, key: str) -> object:
        if key == "id":
            return self.id
        if key == "content":
            return self.content
        if key == "similarity":
            return self.similarity
        if key == "rank_score":
            return self.rank_score
        if key == "date_created":
            return self.date_created
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        yield "id"
        yield "content"
        yield "similarity"
        yield "rank_score"
        yield "date_created"

    def __len__(self) -> int:
        return 5

    def get(self, key: str, default: object = None) -> object:
        try:
            return self[key]
        except KeyError:
            return default
