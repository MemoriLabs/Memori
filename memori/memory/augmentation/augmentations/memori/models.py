from dataclasses import dataclass, field

from memori.memory.augmentation._message import ConversationMessage


@dataclass
class EntityData:
    """Entity metadata structure."""

    id: str | None = None


@dataclass
class ProcessData:
    """Process metadata structure."""

    id: str | None = None


@dataclass
class AttributionData:
    """Attribution metadata structure."""

    entity: EntityData = field(default_factory=EntityData)
    process: ProcessData = field(default_factory=ProcessData)

    def to_dict(self) -> dict[str, object]:
        return {
            "entity": {"id": self.entity.id},
            "process": {"id": self.process.id},
        }


@dataclass
class SessionData:
    """Session metadata structure."""

    id: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id}


@dataclass
class AugmentationInputData:
    attribution: AttributionData
    messages: list[ConversationMessage]
    session: SessionData

    def messages_as_dicts(self) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_dict(self) -> dict[str, object]:
        return {
            "attribution": self.attribution.to_dict(),
            "messages": [m.to_dict() for m in self.messages],
            "session": self.session.to_dict(),
        }
