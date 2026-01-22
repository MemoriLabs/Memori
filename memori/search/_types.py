from dataclasses import dataclass


@dataclass(frozen=True)
class HostedSemanticFact:
    fact_id: str
    fact: str
    similarity: float


@dataclass(frozen=True)
class HostedSemanticFactSet:
    facts: list[HostedSemanticFact]
